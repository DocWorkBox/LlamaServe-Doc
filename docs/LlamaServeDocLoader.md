# LlamaServe-Doc Loader

Configures the managed native `llama-server` process used by LlamaServe-Doc.

## Inputs

- **model**: A `.gguf` model from `ComfyUI/models/LLM`.
- **mmproj**: Optional multimodal projector. Select `None` for text-only models.
- **context_length**: Maximum server context size.
- **gpu_layers**: Exact number of layers offloaded to the GPU. `-1` selects automatic placement.
- **flash_attention**: Enables, disables, or automatically selects Flash Attention.
- **cache_type_k / cache_type_v**: KV-cache precision.
- **port**: Local loopback port used by the managed server.

The Windows NVIDIA CUDA 12 backend is downloaded from the official llama.cpp GitHub release on first use and verified with the release SHA-256 digest.

## Output

- **server_config**: Configuration consumed by `LlamaServe-Doc Generate`.
