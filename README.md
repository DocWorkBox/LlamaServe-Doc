# LlamaServe-Doc

用独立的原生 `llama-server` 在 ComfyUI 中运行 GGUF。它不依赖 LM Studio，也不修改现有的 `llama-cpp_vllm` 节点。

## 安装

```powershell
cd ComfyUI\custom_nodes
git clone https://github.com/DocWorkBox/LlamaServe-Doc.git
```

重启 ComfyUI 后，在 `LlamaServe-Doc` 分类中添加节点。

## 默认演示工作流

仓库自带两个工作流。安装并重启 ComfyUI 后，可以在 `Workflow → Browse Templates → LlamaServe-Doc` 中直接加载：

- `example_workflows/Qwen3.6 H3 Prompt rewrite.json`：纯文本 H3 提示词改写。
- `example_workflows/Qwen2.5 Omni H3 Official Presets.json`：Lightx2v 官方 T2AV / I2AV / L2AV / FL2AV / Ref2AV 输入格式。

该演示默认选择 `Qwen3.6-27B-H3-Prompt-Rewriter-Q4_K_M.gguf`，请把同名模型放入 `ComfyUI/models/LLM/`。工作流中的 `easy showAnything` 仅用于展示生成文本，需要安装 ComfyUI-Easy-Use；不安装时也可以删除这个显示节点，Loader 和 Generate 的推理不受影响。

## 节点

- **LlamaServe-Doc Loader**：选择 `models/LLM` 中的主 GGUF 与 `mmproj`，设置上下文、GPU 层数、Flash Attention 和 KV Cache。llama-server 端口在运行时自动选择并复用，本地媒体白名单自动使用 ComfyUI 根目录，两者都不需要手工填写。
- **LlamaServe-Doc H3 Omni Generate**：推荐的合并节点。内置 Lightx2v 五种模式预设、官方动态参考输入、llama-server 生成，并输出 Director `groups`。
- **LlamaServe-Doc H3 Omni Preset**：旧版拆分式预设节点，为已有工作流继续保留。
- **LlamaServe-Doc Generate**：发送 llama.cpp OpenAI 兼容的流式请求，输出文本和性能 JSON。
- `stop_server_after_generate=false`：默认保留服务器，连续执行时无需重新加载模型，速度最快。
- `stop_server_after_generate=true`：生成结束或报错后停止本节点创建的服务器并释放显存。
- `idle_timeout_minutes=5`：保留服务器时，空闲达到指定分钟数后自动停止并释放显存；设为 `0` 可禁用。
- 点击 ComfyUI 自带的“卸载模型”或“卸载模型并清理执行缓存”时，也会同步停止本节点创建的 llama-server。

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

## Qwen2.5-Omni H3 多模态用法

把以下文件放入 `ComfyUI/models/LLM/`：

```text
Qwen2.5-Omni-7B-H3-Prompt-Rewriter-Q4_K_M.gguf
mmproj-Qwen2.5-Omni-7B-F16.gguf
```

模型可以从 [ModelScope GGUF 仓库](https://modelscope.cn/models/zhaoke1006/Qwen2.5-Omni-7B-H3-Prompt-Rewriter-GGUF) 下载。Loader 中同时选择主 GGUF 和对应的 F16 `mmproj`，建议从以下配置开始：

| 参数 | 起始值 |
| --- | --- |
| context_length | 32768 |
| gpu_layers | -1 |
| flash_attention | on |
| cache_type_k / cache_type_v | q8_0 / q8_0 |
| reasoning | off |

### 官方五种模式预设

推荐直接使用 `LlamaServe-Doc H3 Omni Generate`：只需连接 Loader 的 `server_config`，选择模式并填写 `raw_prompt`。它已经合并旧版 Preset 与 Generate，不再需要在两者之间连接 system/user/media 三条线。节点内置上游仓库的 Base 与 Ref2AV 系统提示词，并按照官方 `infer.py` 组织标题、媒体内容块和最终 Rewrite request。

参考端口与 ComfyUI 官方 `MiniMax H3 Reference to Video` 完全同名、同类型、同 Autogrow 规则：连接最后一个空端口后，会自动出现下一编号。

| 动态输入 | Comfy 类型 | 上限 | 作用 |
| --- | --- | --- | --- |
| `ref_image_0…8` | IMAGE | 9 | `<Picture N>` |
| `ref_video_0…2` | IMAGE 批次（24 fps 视频帧） | 3 | `<Video N>` |
| `ref_video_audio_0…2` | AUDIO | 3 | 与同编号 `ref_video_N` 配对的音轨 |
| `ref_audio_0…2` | AUDIO | 3 | 独立 `<Audio N>` |

顺序也遵循官方节点：先全部图片；然后逐个视频，每个视频若有同编号音轨则先放音频、再放视频；最后放独立音频。孤立的 `ref_video_audio_N`（没有对应 `ref_video_N`）不会进入请求。

| 模式 | Media 输入 | 参考作用 | 允许画幅 |
| --- | --- | --- | --- |
| T2AV | 不连接 Media | 纯文本生成 | adaptive、21:9、16:9、4:3、1:1、3:4、9:16 |
| I2AV | 恰好 1 张 image | 精确首帧 | 同上 |
| L2AV | 恰好 1 张 image | 精确尾帧 | 同上 |
| FL2AV | 按顺序连接 2 张 image | 精确首帧、精确尾帧 | 同上 |
| Ref2AV | 至少 1 个有序 image / video / audio，可混合 | 主体、构图、动作、节奏、声音等完整参考 | 仅 16:9、9:16 |

所有模式的 `duration` 都是 4–15 秒整数。节点会自动映射到 MiniMax-H3 合法的 `17*n+5` 帧网格，并把两位小数的有效时长写进模型输入。

Ref2AV 会按媒体类型分别编号：image 为 `<Picture N>`、video 为 `<Video N>`、audio 为 `<Audio N>`。`raw_prompt` 必须提到每一个实际生效的标签，也不能提到不存在的标签。例如一张图片、一个带配套音轨的视频、一个独立音频会对应 `<Picture 1>`、`<Audio 1>`、`<Video 1>`、`<Audio 2>`。

在 `raw_prompt` 中输入 `@` 会弹出当前节点已经连接且实际生效的媒体列表；可继续输入“图片 / 视频 / 音频”过滤，使用方向键选择并按 Enter / Tab，或直接点击。节点会插入模型要求的官方标签，因此不需要手工计算编号。为兼容已有工作流，中文 `图片1`、`视频1`、`音频1` 也会在执行前转换为对应官方标签；同类参考只有一个时，未编号的“图片 / 视频 / 音频”也可自动对应第 1 个。

```text
Use <Picture 1> for the protagonist, <Video 1> for camera rhythm, and <Audio 1> for the soundtrack.
```

Ref2AV LoRA 最多接受 9 张图片、3 个视频、3 个音频、合计 12 个参考资产。视频帧批次不会自动带入音轨；需要保留同一视频的声音时，把对应 AUDIO 接到同编号 `ref_video_audio_N`，独立声音参考则接 `ref_audio_N`。

### Director groups 输出

合并节点的 `groups` 输出类型为 `MMX_DIR_GROUP`，可直接连接 `MiniMaxH3Director`：

- T2AV / I2AV / L2AV / FL2AV：连接 Director 的 `i2v_groups`。
- Ref2AV：连接 Director 的 `r2v_groups`。

组内 prompt 使用 LoRA 实际生成的增强提示词，而不是原始短提示词；引用的 IMAGE/AUDIO 数据会以 Director 插件当前使用的 `version=1`、`family`、`kind`、`ref_images`、`ref_videos`、`ref_video_audios`、`ref_audios` 结构输出。若要把多个片段组成导演台批次，可继续连接 `MiniMax H3 Director Groups Combine`。

节点执行时会把 Comfy IMAGE/AUDIO 临时转换为 llama.cpp 可读取的 PNG、MP4、WAV，推理完成或报错后自动清理本次临时目录。视频输入需要 FFmpeg。

Base 模式输出三个字段：`integrated_multimodal_description`、`overall_soundscape`、`non_diegetic_music`。Ref2AV 输出六个区段：`subject_definitions`、`summary`、`retention_analysis`、`detailed_description`、`overall_soundscape`、`non_diegetic_music`。

Loader 内部仍会把 ComfyUI 根目录作为 llama-server 的 `--media-path`。这是 llama-server 读取本地 `file://` 图片、视频和音频时必须使用的安全白名单，不是模型参数，也无需用户调整。合并节点会把 ComfyUI 的 IMAGE/AUDIO 输入临时保存到该目录下，再以相对路径发送给后端；因此 Loader 界面不再显示 `media_root`。

若安装了 `comfyui_lg_hotreload`，请在其热加载配置中把 `ComfyUI-LlamaServer` 加入排除列表。旧版 HotReload 会把 ComfyUI V3 的 Combo 类型字符串 `COMBO` 错当成选项数组，导致下拉菜单在热更新后显示为 `C / O / M / B / O`。这属于 HotReload 的 V3 兼容问题，正常重启 ComfyUI 加载本节点不受影响。

### FFmpeg

原生视频输入需要系统能够执行 `ffmpeg` 和 `ffprobe`：

```powershell
ffmpeg -version
ffprobe -version
```

如果命令不存在，请安装 FFmpeg 并把其 `bin` 目录加入系统 `PATH`，然后重启 ComfyUI。图片和独立 WAV/FLAC/MP3 输入不依赖视频解码，但视频文件必须使用带视频支持的 llama.cpp 后端及 FFmpeg。

## 后端下载

第一次执行 Loader + Generate 时，节点从 `ggml-org/llama.cpp` 的 GitHub 最新发行版下载 Windows CUDA 12 后端和 CUDA runtime。每个压缩包都使用 GitHub Release API 返回的 SHA-256 摘要校验后才会安装到本插件的 `runtime/`。后续执行直接复用，不会重复下载。

后端日志位于 `logs/`。节点运行时会自动选择可用的本机端口，并且只会停止由当前节点创建的 llama-server，不会接管或终止外部进程。

## 开发验证

```powershell
python -m unittest discover -s tests -v
```

测试覆盖后端资产选择与 SHA-256 校验、安全解压、服务器复用与停止、流式响应解析、中断处理以及节点注册元数据。

## 中断

ComfyUI 的“中断当前任务”会在服务器启动等待和流式生成期间被检查。中断后若服务器被设置为常驻，将从中断时重新计算空闲超时；也可以直接使用 ComfyUI 自带的显存清理功能立即停止本节点创建的 llama-server。

## 目录

将模型放在：

```text
ComfyUI/models/LLM/*.gguf
```

多模态模型必须选择与主模型匹配的 `mmproj-*.gguf`；纯文本模型继续选择 `None`。
