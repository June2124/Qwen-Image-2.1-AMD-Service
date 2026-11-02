# Qwen-Image-2.1-Turbo AMD FastAPI 服务

本服务通过带鉴权的轻量 FastAPI 接口，提供官方 ComfyUI 分拆权重版 Qwen-Image-2.1-Turbo 工作流。当前配置面向 AMD Radeon AI PRO R9700（`gfx1201`），使用单张 32 GB 显存的 GPU。

Turbo 采样始终采用模型检查点的官方配置：Euler 采样器、八步采样、CFG `1.0`，以及手动指定的 sigma 序列：
`1, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0`。
为保持 API 兼容性，服务仍会解析调用方传入的 `steps`、`guidance_scale` 和 `cfg`，但这些值不会改变实际采样设置。

## 输入图像的角色

图像编辑接口明确区分以下两种角色，避免仅用一个图像数组时混淆主图与参考图：

- `canvas_image`：必填，且只能有一张，作为待编辑的主图，始终以 `<image1>` 传给 Qwen。
- `reference_images`：可选，可提供零至九张素材、风格或物体参考图，按列表顺序对应 `<image2>` 至 `<image10>`。

编辑提示词可以直接引用这些图像编号。文生图接口不接受输入图像。

输出尺寸由图像角色决定，调用方不能自行配置：

- 文生图始终输出 `640x640`。
- 图像编辑严格保留主图的原始尺寸。服务内部会通过复制边缘像素的方式，在右侧和底部补齐，使尺寸达到 32 像素的整数倍以供 Qwen 处理，再将生成的 PNG 裁剪回主图的原始尺寸。
- 参考图单独缩小，默认最长边不超过 640 像素，不会决定或拉伸输出画布的尺寸。

## 运行机制

- FastAPI 进程常驻，但不加载模型权重。
- 第一次推理时按需启动独立的 ComfyUI 子进程。
- 请求串行执行，本服务不会在 GPU 上同时运行两次生成任务。
- 达到配置的空闲时长后，服务会终止 ComfyUI 子进程，释放其内存和显存。设置 `parameters.close_model=true` 后，会在收集到响应图像后立即释放。
- 推理失败或超时也会终止子进程，使下次请求从干净的运行环境开始。

ComfyUI 固定使用版本 `1d2ea2948d33dfda4d7cfe58c6d234968aa62cf8`。构建 Docker 镜像时，会应用来自尚未合并的上游 ComfyUI PR 16603 的补丁，文件为 `patches/comfyui-pr-16603-qwen-vl-linear.patch`。该补丁将 Qwen-VL 图像块嵌入层的 Conv3D 替换为等价的线性运算，以避开 `gfx1201` 上非常缓慢的 AMD `slow_conv3d` 路径，同时保持模型状态字典的键名和数学运算语义不变。

## 模型

部署使用以下模型文件：

- `diffusion_models/qwen_image_2.1_turbo_bf16.safetensors`
- `text_encoders/qwen3vl_8b_int8_convrot.safetensors`
- `vae/qwen_image_2.1_vae_bf16.safetensors`

扩散模型特意使用 BF16 版本。INT8 ConvRot 扩散模型检查点在类似的 `gfx1201` ROCm 环境中曾生成无效图像或黑图。文本编码器仍使用 INT8，配合 ComfyUI 的模型卸载机制，使完整工作流可以在 32 GB 显存下运行。

在此 `gfx1201` 主机上，服务禁用了 `comfy-kitchen` 的 HIP 后端，改用其 eager 回退实现。同时禁用 ComfyUI 的异步卸载和锁页内存，因为两者组合曾在 INT8 文本编码器加载后立即触发 GPU 页错误。常规模型卸载仍然启用。

运行以下脚本从 ModelScope 下载模型，支持断点续传并校验文件大小：

```bash
./download-models.sh
```

## 构建与运行

准备固定版本的 ComfyUI 源码、构建镜像并启动容器：

```bash
export QWEN_API_KEY='replace-me'
./run-service-amd.sh
```

对于无法访问 GitHub 的主机，`scripts/prepare-comfyui.sh` 支持通过 `COMFYUI_ARCHIVE=/path/to/archive.tar.gz` 指定本地源码压缩包。容器默认映射到宿主机的 `18036` 端口。主要环境变量如下：

| 环境变量 | 默认值 |
| --- | --- |
| `QWEN_IDLE_UNLOAD_SECONDS` | `60` |
| `QWEN_INFERENCE_TIMEOUT` | `1800` |
| `COMFY_STARTUP_TIMEOUT` | `300` |
| `COMFY_RESERVE_VRAM_GB` | `1.0` |
| `QWEN_CACHE_DEVICE` | `auto` |
| `QWEN_CACHE_DTYPE` | `default` |
| `QWEN_REFERENCE_MAX_SIDE` | `640` |
| `COMFY_KITCHEN_DISABLE_HIP` | `1` |

请勿在此 AMD 部署中启用实验性的 `int8` Qwen 缓存。多参考图编辑使用该缓存数据类型时，已复现紫色、绿色碎片状损坏图像；相同输入使用常规缓存路径则能生成正常图像。

## API 接口

接口列表：

- `POST /v1/qwen-image/text-to-image`：文生图。
- `POST /v1/qwen-image/image-to-image`：图像编辑。
- `POST /v1/qwen-image/unload`：卸载模型。
- `GET /health`：健康检查。

图像编辑请求示例：

```json
{
  "model": "qwen-image-2.1",
  "input": {
    "prompt": "Keep <image1> unchanged except place the bin from <image2> on the floor.",
    "negative_prompt": "floating objects, stickers, distorted perspective",
    "canvas_image": "data:image/png;base64,...",
    "reference_images": ["data:image/png;base64,..."]
  },
  "parameters": {
    "steps": 50,
    "guidance_scale": 4.0,
    "seed": 1234
  }
}
```

旧版请求参数 `size`、`width` 和 `height` 会被忽略。上述示例特意传入非 Turbo 的采样值，用于展示向后兼容行为：响应会同时报告请求传入的值，以及实际生效的固定值 `steps=8`、`guidance_scale=1.0`。

运行工作流单元测试：

```bash
python -m unittest discover -s tests -v
```

## 许可证说明

Qwen-Image-2.1-Turbo 按照 Qwen Research License 发布。其已公布的条款将用途限制为非商业研究，其他用途需另行获得许可。ComfyUI 和各项随附依赖保留各自的许可证。在生产环境或商业场景中使用前，请核查所有适用条款。
