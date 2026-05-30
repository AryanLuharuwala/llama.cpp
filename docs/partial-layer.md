# Partial-layer load (pipeline-parallel stages)

The `llama_model_params::layer_range_lo` / `layer_range_hi` fields let a single
model load materialise only a contiguous slice of transformer layers `[lo, hi)`.
This is the building block for **pipeline-parallel (PP)** inference, where a model
is split across several nodes and each node owns one *stage* of consecutive
layers.

By default both fields are `0`, which loads the full model and behaves exactly
like an unmodified load. Set `layer_range_hi > layer_range_lo` to enable a
partial load.

## Ownership contract

For a model with `n_layer` transformer blocks, a stage that owns layers
`[lo, hi)`:

| Resource                | Owned by stage when            |
|-------------------------|--------------------------------|
| `token_embd`            | `lo == 0` (first stage)        |
| `blk.<i>.*` for `i`     | `lo <= i < hi`                 |
| `output_norm`, `output` | `hi == n_layer` (last stage)   |

Weights outside the owned set are never allocated, so each node only needs
enough memory/VRAM for its own slice plus (on the boundary stages) the
embedding / LM-head tensors.

## Driving a stage

Activations flow between stages as raw hidden states carried in
`llama_batch.embd`:

* **First stage** (`lo == 0`): drive it normally with token ids. It converts
  tokens to activations via `token_embd`, runs layers `[0, hi)`, and exposes the
  hidden state of layer `hi - 1` as the batch embeddings output.
* **Middle / last stages** (`lo > 0`): do **not** supply token ids. Instead,
  copy the previous stage's output hidden state into `llama_batch.embd`
  (shape `[n_embd, n_tokens]`). These activations become the input to layer
  `lo`.
* **Last stage** (`hi == n_layer`): after running its layers it applies
  `output_norm` and the LM head, producing logits (or pooled embeddings) just
  like a full model.

A minimal three-node pipeline for a 32-layer model:

```c
// node 0: layers [0, 11)  — owns token_embd, driven with tokens
mparams.layer_range_lo = 0;
mparams.layer_range_hi = 11;

// node 1: layers [11, 22) — driven with node 0's hidden state via batch.embd
mparams.layer_range_lo = 11;
mparams.layer_range_hi = 22;

// node 2: layers [22, 32) — owns output_norm/output, emits logits
mparams.layer_range_lo = 22;
mparams.layer_range_hi = 32;
```

Each node calls `llama_decode` on its own context; the orchestrator is
responsible for moving the hidden-state tensor from one stage's output
embeddings into the next stage's `llama_batch.embd`.

## Notes and limitations

* The contract above is implemented for the LLaMA-family graph builder. Other
  architectures fall back to a full load unless they also honour
  `llama_hparams::is_owned_layer()` in their graph and tensor-creation paths.
* Misconfiguration (`hi <= lo`, or a negative `lo`) is ignored and silently
  falls back to a full-model load.
* KV cache, sampling and batching are per-stage and unchanged; only the set of
  materialised weights and the graph's input/output endpoints differ.
