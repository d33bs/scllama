# scllama

Using llama vision models to understand single-cell images for profiling.

## Development

1. Install `llama.cpp` using releases or `brew` (e.g. `brew install llama.cpp`)
1. Download models and run `llama.cpp server` using `llama-server -hf unsloth/gemma-3-4b-it-GGUF:Q4_K_XL` (note: this will download models and could take some time to complete).
1. Run notebooks under `src/scllama`.
