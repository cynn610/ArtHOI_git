# 第三方代码快照

本代码包保留了运行 pipeline 所需的第三方源码和 ArtHOI 本地 wrapper，但去除了各自 `.git/`、权重、缓存、编译目录与历史输出。

| 目录 | Upstream | 整理时 commit |
|---|---|---|
| DiffuEraser | `https://github.com/lixiaowen-xw/diffueraser.git` | `8e6f279ac7531e27ad1849c6f8dab5372a8597e7` |
| Video-Depth-Anything | `https://github.com/DepthAnything/Video-Depth-Anything.git` | `4f5ae23172ba60fd7bc11ef671cca678842c7072` |
| WiLoR | `https://github.com/rolpotamias/WiLoR.git` | `fcb911312a38fa8badd30d9656a167485d61b8f9` |
| CoTracker | `https://github.com/facebookresearch/co-tracker.git` | `82e02e8029753ad4ef13cf06be7f4fc5facdda4d` |
| FoundationPose | `https://github.com/NVlabs/FoundationPose.git` | `a1b694b83e633c2cb6115b9063d940a687759392` |
| PyTorch3D | `https://github.com/facebookresearch/pytorch3d.git` | `33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba` |
| SAM2 | `https://github.com/facebookresearch/sam2.git` | `2b90b9f5ceec907a1c18123530e92e794ad901a4` |
| UniDepth | `https://github.com/lpiccinelli-eth/unidepth.git` | `8d8cfe4c7ee15297099983607febf0d4f32eb3d6` |
| PartField | `https://github.com/nv-tlabs/PartField` | 本地快照；原目录没有 Git metadata |

迁移包直接使用这些快照，避免新服务器重新 clone 后丢失 `app_hrse.py`、`metric_align.py`、`WiLoR_ArtHOI.py`、PartField separation 等本地适配。
