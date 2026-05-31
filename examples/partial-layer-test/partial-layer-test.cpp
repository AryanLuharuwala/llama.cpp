// partial-layer-test: numerical correctness check for the partial-layer /
// pipeline-parallel (PP) split exposed via llama_model_params::layer_range_lo /
// layer_range_hi (see docs/partial-layer.md and include/llama.h).
//
// The program loads the same model twice and proves that splitting the
// transformer stack into two stages produces the same final-token logits as a
// single full-model run.
//
//   1. FULL run   : load 0/0 (full model), decode a fixed prompt, capture the
//                   last-token logits.
//   2. SPLIT run  : stage-0 owns layers [0, mid) and is driven with tokens; its
//                   per-token hidden state (the raw activation after layer
//                   mid-1, exposed as "embeddings") is handed to stage-1, which
//                   owns layers [mid, n_layer), is driven via llama_batch.embd,
//                   and emits the final logits.
//
// PASS if the FULL and SPLIT last-token logits agree to within a small
// tolerance and select the same argmax token.
//
// Two-stage hand-off mechanism (what this test actually uses):
//   * Stage-0 context is created with embeddings=true. Every prompt position is
//     marked as an output (batch.logits[i] = 1) so that llama_get_embeddings_ith
//     returns the carried hidden state for each position. For a non-last stage
//     the graph builder emits the layer (mid-1) hidden state as result_embd
//     (no output_norm, no LM head) -- exactly the activation that must flow to
//     the next stage.
//   * Stage-1 context is driven with a token-less batch built via
//     llama_batch_init(n_tokens, n_embd, 1): batch.token is left NULL and the
//     per-position hidden states are copied into batch.embd ([n_embd, n_tokens]).
//     These become the input to layer `mid`. Being the last stage it applies
//     output_norm + LM head and produces logits read with llama_get_logits_ith.
//
// API gap notes:
//   * The public API has no dedicated "carry hidden state between stages" call.
//     The hand-off reuses the embeddings-output path (llama_get_embeddings_ith)
//     for the stage output and the llama_batch.embd input path for the next
//     stage's input. This works because a non-last partial stage publishes its
//     last-layer hidden state as result_embd, but it does require the caller to
//     (a) enable embeddings mode on every non-last stage and (b) flag every
//     position as an output to retrieve all per-position activations.
//   * The KV cache on each stage is still sized for all n_layer (documented
//     limitation), so the split run uses the same n_ctx as the full run.

#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::string get_model_path(int argc, char ** argv) {
    if (argc > 1 && argv[1][0] != '\0') {
        return std::string(argv[1]);
    }
    const char * env = getenv("LLAMA_TEST_MODEL");
    if (env != nullptr && env[0] != '\0') {
        return std::string(env);
    }
    return std::string();
}

// Decode the full (un-split) model and copy the last-token logits out.
static bool run_full(const std::string & model_path,
                     const std::vector<llama_token> & tokens,
                     std::vector<float> & logits_out,
                     int & n_vocab_out) {
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0; // CPU: deterministic + no VRAM assumptions

    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (model == nullptr) {
        fprintf(stderr, "[full] failed to load model\n");
        return false;
    }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    n_vocab_out = n_vocab;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = (uint32_t) tokens.size();
    cparams.n_batch = (uint32_t) tokens.size();
    cparams.no_perf = true;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "[full] failed to create context\n");
        llama_model_free(model);
        return false;
    }

    bool ok = true;

    // token batch; only the last token needs logits.
    llama_batch batch = llama_batch_get_one(const_cast<llama_token *>(tokens.data()),
                                            (int32_t) tokens.size());
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "[full] decode failed\n");
        ok = false;
    }

    if (ok) {
        const float * logits = llama_get_logits_ith(ctx, (int32_t) tokens.size() - 1);
        if (logits == nullptr) {
            fprintf(stderr, "[full] no logits\n");
            ok = false;
        } else {
            logits_out.assign(logits, logits + n_vocab);
        }
    }

    llama_free(ctx);
    llama_model_free(model);
    return ok;
}

// Stage-0: load [0, mid), drive with tokens, return per-position hidden states.
static bool run_stage0(const std::string & model_path,
                       const std::vector<llama_token> & tokens,
                       int mid,
                       int & n_embd_out,
                       std::vector<float> & hidden_out) {
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers    = 0;
    mparams.layer_range_lo  = 0;
    mparams.layer_range_hi  = mid;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (model == nullptr) {
        fprintf(stderr, "[stage0] failed to load model\n");
        return false;
    }

    const int n_embd = llama_model_n_embd(model);
    n_embd_out = n_embd;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx      = (uint32_t) tokens.size();
    cparams.n_batch    = (uint32_t) tokens.size();
    cparams.no_perf    = true;
    cparams.embeddings = true; // expose the carried hidden state

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "[stage0] failed to create context\n");
        llama_model_free(model);
        return false;
    }

    bool ok = true;

    const int n_tokens = (int) tokens.size();

    // build a token batch and request output (embeddings) for EVERY position,
    // since stage-1 attends across all positions.
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    batch.n_tokens = n_tokens;
    for (int i = 0; i < n_tokens; ++i) {
        batch.token[i]     = tokens[i];
        batch.pos[i]       = i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = 1; // output this position's hidden state
    }

    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "[stage0] decode failed\n");
        ok = false;
    }

    if (ok) {
        hidden_out.resize((size_t) n_tokens * n_embd);
        for (int i = 0; i < n_tokens && ok; ++i) {
            const float * e = llama_get_embeddings_ith(ctx, i);
            if (e == nullptr) {
                fprintf(stderr, "[stage0] no embeddings for token %d\n", i);
                ok = false;
                break;
            }
            memcpy(hidden_out.data() + (size_t) i * n_embd, e, sizeof(float) * n_embd);
        }
    }

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    return ok;
}

// Stage-1: load [mid, n_layer), drive via batch.embd, return last-token logits.
static bool run_stage1(const std::string & model_path,
                       int mid,
                       int n_layer,
                       int n_tokens,
                       int n_embd,
                       const std::vector<float> & hidden_in,
                       std::vector<float> & logits_out,
                       int & n_vocab_out) {
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers   = 0;
    mparams.layer_range_lo = mid;
    mparams.layer_range_hi = n_layer;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (model == nullptr) {
        fprintf(stderr, "[stage1] failed to load model\n");
        return false;
    }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    n_vocab_out = n_vocab;

    if (llama_model_n_embd(model) != n_embd) {
        fprintf(stderr, "[stage1] n_embd mismatch (%d vs %d)\n",
                llama_model_n_embd(model), n_embd);
        llama_model_free(model);
        return false;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = (uint32_t) n_tokens;
    cparams.n_batch = (uint32_t) n_tokens;
    cparams.no_perf = true;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "[stage1] failed to create context\n");
        llama_model_free(model);
        return false;
    }

    bool ok = true;

    // token-less batch carrying the stage-0 hidden states via batch.embd.
    llama_batch batch = llama_batch_init(n_tokens, n_embd, 1);
    batch.n_tokens = n_tokens;
    memcpy(batch.embd, hidden_in.data(), sizeof(float) * (size_t) n_tokens * n_embd);
    for (int i = 0; i < n_tokens; ++i) {
        batch.pos[i]       = i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = (i == n_tokens - 1) ? 1 : 0;
    }

    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "[stage1] decode failed\n");
        ok = false;
    }

    if (ok) {
        const float * logits = llama_get_logits_ith(ctx, n_tokens - 1);
        if (logits == nullptr) {
            fprintf(stderr, "[stage1] no logits\n");
            ok = false;
        } else {
            logits_out.assign(logits, logits + n_vocab);
        }
    }

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    return ok;
}

static int argmax(const std::vector<float> & v) {
    int   best_i = 0;
    float best_v = v.empty() ? 0.0f : v[0];
    for (int i = 1; i < (int) v.size(); ++i) {
        if (v[i] > best_v) {
            best_v = v[i];
            best_i = i;
        }
    }
    return best_i;
}

int main(int argc, char ** argv) {
    const std::string model_path = get_model_path(argc, argv);
    if (model_path.empty()) {
        fprintf(stderr,
                "usage: %s <model.gguf>   (or set LLAMA_TEST_MODEL)\n"
                "  proves the layer_range_lo/hi pipeline split is numerically correct.\n",
                argv[0]);
        return 2;
    }

    // tolerance for the FULL-vs-SPLIT last-token logit comparison.
    // CPU greedy decode of the same weights should match to within rounding of
    // the extra round-trip of the hidden state through f32 batch.embd.
    const float tol = 1e-3f;

    ggml_backend_load_all();
    llama_log_set(nullptr, nullptr); // keep test output readable; comment out to debug

    // a short fixed token sequence (avoids vocab assumptions; valid for any
    // model whose vocab has at least these ids).
    const std::vector<llama_token> tokens = { 1, 2, 3, 4, 5, 6, 7, 8 };

    // discover n_layer with a real (tensor-loaded) model. vocab_only must NOT
    // be set: in vocab-only mode the loader skips tensors and reports n_layer
    // as 0, which would make the split impossible. Start from defaults and only
    // pin n_gpu_layers=0 so the probe matches the CPU loads used below.
    int n_layer = 0;
    {
        llama_model_params mparams = llama_model_default_params();
        mparams.n_gpu_layers = 0;
        llama_model * m = llama_model_load_from_file(model_path.c_str(), mparams);
        if (m == nullptr) {
            fprintf(stderr, "failed to load model from '%s'\n", model_path.c_str());
            return 1;
        }
        n_layer = llama_model_n_layer(m);
        llama_model_free(m);
    }

    if (n_layer < 2) {
        fprintf(stderr, "model has too few layers (%d) to split\n", n_layer);
        return 1;
    }

    const int mid = n_layer / 2;
    printf("model      : %s\n", model_path.c_str());
    printf("n_layer    : %d\n", n_layer);
    printf("split      : stage0 [0,%d)  stage1 [%d,%d)\n", mid, mid, n_layer);
    printf("n_tokens   : %d\n", (int) tokens.size());

    // ---- FULL run -------------------------------------------------------
    std::vector<float> full_logits;
    int full_vocab = 0;
    if (!run_full(model_path, tokens, full_logits, full_vocab)) {
        fprintf(stderr, "RESULT: FAIL (full run failed)\n");
        return 1;
    }

    // ---- SPLIT run ------------------------------------------------------
    int n_embd = 0;
    std::vector<float> hidden;
    if (!run_stage0(model_path, tokens, mid, n_embd, hidden)) {
        fprintf(stderr, "RESULT: FAIL (stage0 failed)\n");
        return 1;
    }

    std::vector<float> split_logits;
    int split_vocab = 0;
    if (!run_stage1(model_path, mid, n_layer, (int) tokens.size(), n_embd,
                    hidden, split_logits, split_vocab)) {
        fprintf(stderr, "RESULT: FAIL (stage1 failed)\n");
        return 1;
    }

    if (full_vocab != split_vocab ||
        full_logits.size() != split_logits.size() ||
        full_logits.empty()) {
        fprintf(stderr, "RESULT: FAIL (vocab/size mismatch: %d vs %d)\n",
                full_vocab, split_vocab);
        return 1;
    }

    // ---- compare --------------------------------------------------------
    float max_abs_diff = 0.0f;
    for (size_t i = 0; i < full_logits.size(); ++i) {
        const float d = std::fabs(full_logits[i] - split_logits[i]);
        if (d > max_abs_diff) {
            max_abs_diff = d;
        }
    }

    const int full_arg  = argmax(full_logits);
    const int split_arg = argmax(split_logits);

    printf("n_embd     : %d\n", n_embd);
    printf("n_vocab    : %d\n", full_vocab);
    printf("argmax     : full=%d  split=%d\n", full_arg, split_arg);
    printf("full logit : %.6f\n", full_logits[full_arg]);
    printf("split logit: %.6f\n", split_logits[split_arg]);
    printf("max|diff|  : %.6g   (tol=%.6g)\n", max_abs_diff, tol);

    const bool pass = (max_abs_diff <= tol) && (full_arg == split_arg);
    printf("RESULT: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
