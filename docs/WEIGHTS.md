# 模型权重清单

模型权重没有放入本代码目录。下列 SHA-256 来自整理时原工作区已有文件，可用于单独迁移后的完整性检查。

| 模块 | 目标相对路径 | 原大小 | SHA-256 |
|---|---|---:|---|
| SAM2 | `third_party/sam2/checkpoints/sam2.1_hiera_base_plus.pt` | 323,606,802 | `a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5` |
| VDA vits | `third_party/Video-Depth-Anything/checkpoints/video_depth_anything_vits.pth` | 116,440,756 | `13379300b739e659f076a59d52e9801bd8d38c541a7e71f73bbca4dcfb013609` |
| CoTracker online | `third_party/co-tracker/checkpoints/scaled_online.pth` | 101,695,610 | `205d34789f19699d64b22cf93f9b697f15f28d4025240e31532e504109837218` |
| FoundationPose scorer | `third_party/foundationpose/weights/2024-01-11-20-02-45/model_best.pth` | 190,229,389 | `81924d384bf5c26c646ee4783104982ae3d1e049c181c36641b6a7aeae494c26` |
| FoundationPose refiner | `third_party/foundationpose/weights/2023-10-28-18-33-37/model_best.pth` | 68,220,109 | `774700586ddc435d408fc01c9809c43e151232936369dfbea0f0f964ba471d60` |
| WiLoR | `third_party/WiLoR/pretrained_models/wilor_final.ckpt` | 2,564,989,533 | `3e97aafc7dd08d883a4cc5a027df61fdb6fda6136dbd1319405413862ada6bb2` |
| WiLoR detector | `third_party/WiLoR/pretrained_models/detector.pt` | 53,582,271 | `5ef3df44e42d2db52d4ffe91f83a22ce9925e2acc9abebf453f2c5d22e380033` |
| MANO right (WiLoR) | `third_party/WiLoR/mano_data/MANO_RIGHT.pkl` | 3,821,356 | `45d60aa3b27ef9107a7afd4e00808f307fd91111e1cfa35afd5c4a62de264767` |
| MANO right | `third_party/body_models/MANO_RIGHT.pkl` | 3,821,356 | `45d60aa3b27ef9107a7afd4e00808f307fd91111e1cfa35afd5c4a62de264767` |
| MANO left | `third_party/body_models/MANO_LEFT.pkl` | 3,821,391 | `c4022f7083f2ca7c78b2b3d595abbab52debd32b09d372b16923a801f0ea6a30` |

FoundationPose 的两个权重目录还必须包含各自的 `config.yml`。原工作区对应 SHA-256 为：

```text
a79db4de3b95885dd5ae86833b37b8698a75dad81e87d1086cd50b2fcd8dda3f  2024-01-11-20-02-45/config.yml
28a6ba94a33230ee5fc3c51939486281578b0972542bd9e38ca6123e75605686  2023-10-28-18-33-37/config.yml
```

原工作区没有可迁移的完整 DiffuEraser 与 PartField 权重，需要按各自官方说明准备：

```text
third_party/DiffuEraser/weights/stable-diffusion-v1-5/
third_party/DiffuEraser/weights/sd-vae-ft-mse/
third_party/DiffuEraser/weights/diffuEraser/
third_party/DiffuEraser/weights/propainter/
third_party/PartField/model/model_objaverse.ckpt
```

UniDepthV2 使用 `lpiccinelli/unidepth-v2-vitl14` 并在首次运行时下载。完成迁移后运行：

```bash
python scripts/check_environment.py --strict
```
