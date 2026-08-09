# ArtHOI 可迁移完整 Pipeline

这是从当前可用工作区独立整理出的干净代码包。它以单目 RGB 视频为输入，串联视频预处理、手/物体先验、接触推理、物体 4D 运动与最终手–物体联合重建。

本目录不包含数据集、历史重建结果、日志、缓存、模型权重或原项目的 Git 历史。原目录没有被修改。

## 一条命令

先按 [迁移与安装](docs/MIGRATION.md) 配好环境和权重，再运行：

```bash
CUDA_VISIBLE_DEVICES=0 python run_pipeline.py \
  --video /path/to/input.mp4 \
  --seq-name my_sequence \
  --part-cnt 2 \
  --cano-frame 0 \
  --target-reso 960 540
```

预处理阶段会暂停等待人工操作，这是算法设计的一部分：需要在 SAM2 页面标注/检查 human 与 `part_i` masks，并检查 Hunyuan3D 网格及 PartField 分件。每个提示中按 Enter 执行，输入 `s` 跳过已有结果，输入 `exit` 退出；重新运行同一命令即可继续。

## 模型权重下载

官方 [ArtHOI-4D-Reconstruction](https://github.com/hitcs-zikaiwang/ArtHOI-4D-Reconstruction) GitHub 仓库不包含模型权重，也不会在普通 `git clone` 时自动下载权重；多个 `third_party` 项目及其权重需要按官方说明另外准备。本代码包同样有意排除了模型权重，避免将大文件或受许可证约束的文件提交到 Git。

整理时的原工作区本身也没有完整的 DiffuEraser 和 PartField 权重：DiffuEraser 只有 `weights/README.md`，PartField 没有 `model_objaverse.ckpt`。因此这两组权重需要在新、旧服务器上按下面步骤补齐。

以下命令均从本仓库根目录执行：

```bash
cd /path/to/ArtHOI_git

python -m pip install -U huggingface_hub

mkdir -p third_party/DiffuEraser/weights/{diffuEraser,PCM_Weights,stable-diffusion-v1-5,sd-vae-ft-mse,propainter}
mkdir -p third_party/PartField/model
```

### DiffuEraser 主模型

```bash
hf download lixiaowen/diffuEraser \
  --local-dir third_party/DiffuEraser/weights/diffuEraser
```

当前 pipeline 的 DiffuEraser 使用 PCM 2-Step，还必须下载下面这个文件：

```bash
hf download wangfuyun/PCM_Weights \
  sd15/pcm_sd15_smallcfg_2step_converted.safetensors \
  --local-dir third_party/DiffuEraser/weights/PCM_Weights
```

### Stable Diffusion 1.5 与 VAE

下载推理需要的精简版 Stable Diffusion 1.5 文件，约 4 GB。不要直接下载超过 30 GB 的完整训练仓库，除非还需要训练：

```bash
hf download stable-diffusion-v1-5/stable-diffusion-v1-5 \
  --include "feature_extractor/*" "model_index.json" \
            "safety_checker/*" "scheduler/*" \
            "text_encoder/*" "tokenizer/*" \
  --local-dir third_party/DiffuEraser/weights/stable-diffusion-v1-5

hf download stabilityai/sd-vae-ft-mse \
  --local-dir third_party/DiffuEraser/weights/sd-vae-ft-mse
```

### ProPainter

ProPainter 在首次运行且服务器可访问 GitHub 时会尝试自动下载。为了便于离线迁移和提前发现网络问题，建议显式下载：

```bash
curl -L --fail -o third_party/DiffuEraser/weights/propainter/raft-things.pth \
  https://github.com/sczhou/ProPainter/releases/download/v0.1.0/raft-things.pth

curl -L --fail -o third_party/DiffuEraser/weights/propainter/recurrent_flow_completion.pth \
  https://github.com/sczhou/ProPainter/releases/download/v0.1.0/recurrent_flow_completion.pth

curl -L --fail -o third_party/DiffuEraser/weights/propainter/ProPainter.pth \
  https://github.com/sczhou/ProPainter/releases/download/v0.1.0/ProPainter.pth
```

### PartField

```bash
hf download mikaelaangel/partfield-ckpt \
  model_objaverse.ckpt \
  --local-dir third_party/PartField/model
```

如果不使用 PartField，可运行 pipeline 时增加 `--use-2d-mask`，使用 2D mask 进行网格分件，此时不需要下载 PartField 权重。两种分件方法的结果可能不同，因此需要与目标复现实验保持同一设置。

### 权重检查

下载后先显式检查 PCM 和 PartField，再运行完整环境检查：

```bash
test -f third_party/DiffuEraser/weights/PCM_Weights/sd15/pcm_sd15_smallcfg_2step_converted.safetensors
test -f third_party/PartField/model/model_objaverse.ckpt

python scripts/check_environment.py --strict
```

当前严格检查会检查 DiffuEraser 的主模型、Stable Diffusion、VAE、ProPainter 和 PartField，但不会单独检查 PCM 文件，所以不能省略上面的 PCM `test`。SAM2、Video-Depth-Anything、CoTracker、FoundationPose、WiLoR、MANO 等其余权重的放置路径、大小与 SHA-256 见 [完整权重清单](docs/WEIGHTS.md)。MANO 受许可证约束，必须通过 [MANO 官方网站](https://mano.is.tue.mpg.de/) 获取，不应放入公开 Git 仓库。

权重、依赖版本、mask、规范帧和配置都会影响重建结果。要尽可能复现原 pipeline 的效果，应在服务器之间迁移同一批权重并校验哈希；仅使用相同代码不能保证数值结果完全一致。

完整阶段为：

```text
video
  -> preprocess + masks + inpainting + depth + camera
  -> canonical object + WiLoR/MANO hands
  -> MLLM/manual contact
  -> ASR canonical registration
  -> object part separation
  -> per-part 4D motion
  -> hand-object alignment
  -> reconstructed frames and videos
```

## 左手拉回物体

生成配置默认包含：

```yaml
ho_align:
  xyz_pullback_instances: [left]
```

因此 HO alignment 的 pull-back 阶段允许左手沿 XYZ 三个方向移动；未列出的手保持原来的 Z-only 修正。入口会在 ASR 和最终对齐前再次校验这个配置，并默认要求接触文件中存在 `left` 实例，防止迁移后静默退回旧行为。

## 输出

默认所有运行时文件都写到 `workspace/`，不会污染代码：

```text
workspace/
├── sequences/<seq>/                 # 当前输入产生的数据与中间先验
├── configs/<seq>_pNcF.yml           # 序列配置
├── intermediate/<seq>/              # MLLM windows/responses
└── reconstruction/<seq>/
    ├── asr/<seq>/
    ├── canonical_parts/<seq>/
    ├── object_4d/
    └── hoi/<seq>/
        ├── iter_800/data_hand.npy
        └── final_opt/                # 最终逐帧可视化和视频
```

断点运行示例：

```bash
python run_pipeline.py --video input.mp4 --part-cnt 2 --cano-frame 0 \
  --from-stage asr --to-stage object_motion

python run_pipeline.py --list-stages
python run_pipeline.py --video input.mp4 --dry-run
python run_pipeline.py --doctor
```

若不使用视觉大模型 API，可增加 `--contact-mode manual`；已有接触标注则使用 `--contact-mode existing --contact-json /path/ho_contact.json`。

详细资料：

- [Pipeline 各阶段与检查点](docs/PIPELINE.md)
- [服务器迁移与环境安装](docs/MIGRATION.md)
- [权重目录、大小与校验值](docs/WEIGHTS.md)
- [第三方代码版本](docs/THIRD_PARTY.md)

## Citation

```bibtex
@inproceedings{wang2026arthoi,
  title={ArtHOI: Taming Foundation Models for Monocular 4D Reconstruction of Hand-Articulated-Object Interactions},
  author={Wang, Zikai and Zhang, Zhilu and Wang, Yiqing and Li, Hui and Zuo, Wangmeng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2026}
}
```
