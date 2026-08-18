# LlamaServe-Doc Generate

Starts or reuses the configured `llama-server`, sends a streaming chat-completion request, and returns the generated text plus performance data.

## Important inputs

- **system_prompt / user_prompt**: Chat messages sent to the model.
- **reasoning**: `off` disables Qwen thinking output, `auto` uses the model default, and `on` enables thinking.
- **stop_server_after_generate**:
  - `false`: Keep the server and model loaded for fast repeated generation.
  - `true`: Stop the owned server after success or failure and release its model memory.

ComfyUI interruption is checked while the server starts and while response chunks are being received.

## Outputs

- **text**: Generated assistant text.
- **performance_json**: llama.cpp timing data and managed-server state.
