# LlamaServe-Doc

用独立的原生 `llama-server` 在 ComfyUI 中运行 GGUF。它不依赖 LM Studio，也不修改现有的 `llama-cpp_vllm` 节点。

## 安装

```powershell
cd ComfyUI\custom_nodes
git clone https://github.com/DocWorkBox/LlamaServe-Doc.git
```

重启 ComfyUI 后，在 `LlamaServe-Doc` 分类中添加节点。

## 默认演示工作流

仓库自带 `example_workflows/Qwen3.6 H3 Prompt rewrite.json`。安装并重启 ComfyUI 后，可以在 `Workflow → Browse Templates → LlamaServe-Doc` 中直接加载。

该演示默认选择 `Qwen3.6-27B-H3-Prompt-Rewriter-Q4_K_M.gguf`，请把同名模型放入 `ComfyUI/models/LLM/`。工作流中的 `easy showAnything` 仅用于展示生成文本，需要安装 ComfyUI-Easy-Use；不安装时也可以删除这个显示节点，Loader 和 Generate 的推理不受影响。

## 节点

- **LlamaServe-Doc Loader**：选择 `models/LLM` 中的 GGUF，设置上下文、GPU 层数、Flash Attention、KV Cache 和端口。
- **LlamaServe-Doc Generate**：发送 OpenAI 兼容的流式请求，输出文本和性能 JSON。
- `stop_server_after_generate=false`：默认保留服务器，连续执行时无需重新加载模型，速度最快。
- `stop_server_after_generate=true`：生成结束或报错后停止本节点创建的服务器并释放显存。

## 推荐起始参数

针对 16 GB 显存上的 `Qwen3.6-27B-H3-Prompt-Rewriter-Q4_K_M.gguf`：

| 参数 | 起始值 |
| --- | --- |
| context_length | 4096 |
| gpu_layers | 47 |
| flash_attention | on |
| cache_type_k / cache_type_v | q8_0 / q8_0 |
| mmproj | None（纯文本提示词改写不需要） |
| reasoning | off |

`gpu_layers=-1` 表示让 llama.cpp 自动决定。显存不足时先降低 GPU 层数；有余量时再逐步提高。

## 后端下载

第一次执行 Loader + Generate 时，节点从 `ggml-org/llama.cpp` 的 GitHub 最新发行版下载 Windows CUDA 12 后端和 CUDA runtime。每个压缩包都使用 GitHub Release API 返回的 SHA-256 摘要校验后才会安装到本插件的 `runtime/`。后续执行直接复用，不会重复下载。

后端日志位于 `logs/`。若端口被外部程序占用，节点会报错而不会停止不属于自己的进程。

## 开发验证

```powershell
python -m unittest discover -s tests -v
```

测试覆盖后端资产选择与 SHA-256 校验、安全解压、服务器复用与停止、流式响应解析、中断处理以及节点注册元数据。

## 发布到 Comfy Registry

仓库已包含官方 Registry 所需的 `pyproject.toml`、`.comfyignore` 和手动发布工作流 `.github/workflows/publish_action.yml`。

首次注册前：

1. 在 [Comfy Registry](https://registry.comfy.org/) 创建 Publisher，Publisher ID 使用 `DocWorkBox`。Publisher ID 创建后不可更改。
2. 为该 Publisher 创建 Registry API Key。
3. 在 GitHub 仓库的 `Settings → Secrets and variables → Actions` 中新增仓库 Secret：`REGISTRY_ACCESS_TOKEN`。
4. 打开 GitHub `Actions → Publish to Comfy Registry → Run workflow`，手动发布 `1.0.0`。

后续版本按语义化版本更新 `pyproject.toml` 中的 `version`，提交推送后再次手动运行发布工作流。Registry 中的项目 `name` 在首次发布后不可更改。

## 中断

ComfyUI 的“中断当前任务”会在服务器启动等待和流式生成期间被检查。中断生成不会自动关闭常驻服务器；若需要立即释放显存，请打开 `stop_server_after_generate` 后再次执行，或重启 ComfyUI。

## 目录

将模型放在：

```text
ComfyUI/models/LLM/*.gguf
```

多模态模型才需要选择 `mmproj-*.gguf`。本节点当前只发送文本消息，`mmproj` 主要为后续多模态扩展预留。
