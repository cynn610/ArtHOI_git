# 完整 Pipeline 与断点

`run_pipeline.py` 是唯一推荐入口。它用当前 Python 执行主流程，并通过 `conda run` 调用配置的环境；环境名可用 `--main-env`、`--asr-env` 或 `ARTHOI_*_ENV` 调整。

| 阶段 | 主要输入 | 主要输出 | 人工检查 |
|---|---|---|---|
| `preprocess` | RGB video、part 数、规范帧 | frames、masks、depth、camera、canonical mesh、WiLoR/MANO | SAM2、去手视频、部件 mask、mesh/parts |
| `contact_windows` | RGB + metric depth | 连续 RGB-D 拼图 | 抽查 RGB/depth 对齐 |
| `contact_inference` | RGB-D 拼图 | `processed/ho_contact.json` | 检查左右手与接触手指 |
| `asr` | 规范帧 mesh、mask、depth | `pred_pose.npy`、`final_mesh_asr.obj` | 检查规范帧 overlay/IoU |
| `part_separation` | ASR mesh、PartField vmaps、2D masks | `part_i_partdata.npy` | 检查每个部件非空且边界正确 |
| `object_motion` | 规范部件、CoTracker、逐帧 mask/depth | `part_i_best.npy` | 检查长序列漂移 |
| `hoi_alignment` | object 4D、MANO、contact | `data_hand.npy`、最终视频 | 检查接触、穿模、抖动 |

每个成功阶段在 `workspace/.pipeline/<seq>/<stage>.done` 写标记。重复同一命令会跳过成功阶段；`--force` 强制重跑选中范围。若某阶段中途停止，不会写标记，可直接重新运行并在 `make_data.py` 提示中跳过已经完成的子步骤。

## 关键数据契约

```text
<seq>/packed/visuals.npz
<seq>/packed/visuals_inpainting.npz
<seq>/processed/wilor_af/manoparam_fit.slerp.npy
<seq>/processed/ho_contact.json
```

最终对齐还要求 object-motion 目录中每个部件同时存在 `part_i_partdata.npy` 与 `part_i_best.npy`。

## 左手 XYZ pull-back

`hrse/hand_instances.py` 将“手实例 ID”和 MANO handedness 分开。传统视频的实例 ID 是 `left/right`。`hrse/ho_align.py` 读取 `ho_align.xyz_pullback_instances`：列入的实例在 pull-back 阶段使用完整 XYZ residual；其他实例对 residual 与逐帧 translation 都进行 Z-only 梯度约束。

默认入口会检查：

1. YAML 中 `xyz_pullback_instances` 包含 `left`；
2. `ho_contact.json` 中 `appeared` 或 `hand_instances` 包含 `left`。

只有明确不需要左手时才使用 `--allow-missing-left`。

## PartField 与 2D fallback

默认配置设置 `PF_cluster_cnt = max(2 * part_cnt, 2)`，先过分割再由 2D mask 投票。`--use-2d-mask` 会自动跳过预处理中的 PartField，并在 canonical part separation 中强制按 2D mask 切网格；空的 `processed/partseps/` 目录仍会保留以维持统一数据契约。

## MLLM 替代方案

- 默认：`--contact-mode mllm`，需要 `QWEN_API_KEY` 或 OpenAI-compatible relay 配置。
- 人工：`--contact-mode manual`，逐段标注 contact 与 `thumb/index/middle`。
- 已有 JSON：`--contact-mode existing --contact-json ...`。

接触标签是强几何约束。错误的长接触区间可能把手拉向错误部件，必须在最终优化前检查。
