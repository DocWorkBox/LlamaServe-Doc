# LlamaServe-Doc Loader

Configures the managed native `llama-server` process used by LlamaServe-Doc.

## Inputs

- **model**: A `.gguf` model from `ComfyUI/models/LLM`.
- **mmproj**: Optional multimodal projector. Select `None` for text-only models.
- **context_length**: Maximum server context size.
- **gpu_layers**: Exact number of layers offloaded to the GPU. `-1` selects automatic placement.
- **flash_attention**: Enables, disables, or automatically selects Flash Attention.
- **cache_type_k / cache_type_v**: KV-cache precision.
- **backend**: `auto` selects the recommended runtime for the current operating system. Advanced users can choose CUDA, Vulkan, Metal, or CPU explicitly.

On first use, the node obtains a matching official llama.cpp runtime. `auto` uses CUDA 12 on Windows x64 and Metal on Apple Silicon. On Linux x64 or ARM64 it first looks for a complete CUDA build toolchain (`nvcc`, CMake, and Git); when found, it checks out a recent official llama.cpp nightly tag, builds `llama-server` with CUDA, and caches the resulting runtime. Without that toolchain, Linux falls back to Vulkan when `vulkaninfo` is available and otherwise to CPU.

ComfyUI/PyTorch CUDA support alone does not guarantee that the CUDA Toolkit compiler (`nvcc`) is installed. Prebuilt runtimes are verified with their GitHub release SHA-256 digest. Users who already have a CUDA-enabled `llama-server` can place it on `PATH` or set `LLAMASERVE_DOC_SERVER` to its absolute path; this takes priority over download and compilation.

## Output

- **server_config**: Configuration consumed by `LlamaServe-Doc Generate`.
