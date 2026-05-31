# dependencies

find_package(Threads REQUIRED)

# third-party

# ...

# flags

llama_add_compile_flags()

# options

# Optional standalone test for the partial-layer / pipeline-parallel split
# (layer_range_lo/hi). Off by default so it never disturbs the normal build;
# enable with -DLLAMA_BUILD_PARTIAL_TEST=ON.
option(LLAMA_BUILD_PARTIAL_TEST "llama: build the partial-layer pipeline test" OFF)

# examples

if (EMSCRIPTEN)
else()
    if (LLAMA_BUILD_PARTIAL_TEST)
        add_subdirectory(partial-layer-test)
    endif()
    add_subdirectory(batched)
    add_subdirectory(debug)
    add_subdirectory(embedding)
    add_subdirectory(eval-callback)

    add_subdirectory(gguf-hash)
    add_subdirectory(gguf)
    add_subdirectory(idle)
    add_subdirectory(lookahead)
    add_subdirectory(lookup)
    add_subdirectory(parallel)
    add_subdirectory(passkey)
    add_subdirectory(retrieval)
    add_subdirectory(simple)
    add_subdirectory(simple-chat)
    add_subdirectory(speculative)
    add_subdirectory(speculative-simple)
    add_subdirectory(gen-docs)
    add_subdirectory(training)
    add_subdirectory(diffusion)
    if (NOT GGML_BACKEND_DL)
        add_subdirectory(convert-llama2c-to-ggml)
        # these examples use the backends directly and cannot be built with dynamic loading
        if (GGML_SYCL)
            add_subdirectory(sycl)
        endif()
    endif()
endif()
