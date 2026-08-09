# 服务器迁移与安装

## 1. 传输代码

整个 `ArtHOI_git` 可以直接传输；运行数据与权重已经由 `.gitignore` 隔离：

```bash
rsync -a --info=progress2 ArtHOI_git/ user@server:/path/ArtHOI_git/
```

不要从旧项目再复制 `data/`、`logs/`、`outputs/`、`workspace/` 或任何历史 reconstruction 目录。权重按 [WEIGHTS.md](WEIGHTS.md) 单独传输，便于校验和管理大文件。

## 2. 参考环境

系统侧需要 NVIDIA driver、与 PyTorch 匹配的 CUDA toolkit（编译扩展时必须有 `nvcc`）、`ffmpeg/ffprobe`、C/C++ compiler、CMake 和 Ninja。

已验证工作区的核心版本为：

```text
Python       3.10.0
PyTorch      2.2.0+cu121
torchvision  0.17.0+cu121
torchaudio   2.2.0+cu121
NumPy        1.26.4
OpenCV       4.11.0
```

这套研究代码的第三方仓库声明存在互相冲突的依赖（尤其是 torch 和 NumPy）。不要直接让各第三方项目升级依赖；以主环境版本为准，局部包使用 `--no-deps` 安装。

## 3. 主环境

```bash
mamba create -n arthoi python=3.10.0 -y
mamba activate arthoi

python -m pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-pipeline.txt
python -m pip install -e . --no-build-isolation
```

本地第三方代码按当前快照安装：

```bash
python -m pip install -e third_party/co-tracker --no-deps
python -m pip install -e third_party/unidepth --no-deps
SAM2_BUILD_CUDA=0 python -m pip install -e third_party/sam2 --no-deps

FORCE_CUDA=1 python -m pip install -e third_party/pytorch3d \
  --no-build-isolation --no-deps
```

再安装/编译 `nvdiffrast==0.3.3`、`manopth==0.0.1` 和 `chumpy==0.71`。它们没有被放进普通 requirements，因为必须与 PyTorch/CUDA 匹配。参考命令：

```bash
python -m pip install --no-build-isolation \
  'git+https://github.com/NVlabs/nvdiffrast.git@729261dc64c4241ea36efda84fbf532cc8b425b8'
python -m pip install \
  'git+https://github.com/hassony2/manopth.git@4f1dcad1201ff1bfca6e065a85f0e3456e1aa32b'
python -m pip install --no-build-isolation \
  'git+https://github.com/mattloper/chumpy.git@51d5afd92a8ded3637553be8cef41f328a1c863a'
```

FoundationPose 还需要 CUDA toolkit、CMake、Eigen、Boost、pybind11 与 `mycpp` 扩展。安装系统/conda 编译依赖后，在 `third_party/foundationpose/` 运行其 `build_all_conda.sh`。若单独建环境，通过 `--asr-env <name>` 指定；若合并在主环境，使用 `--asr-env arthoi`。

## 4. 权重和 API

```bash
cp conf/api_keys.env.example conf/api_keys.env
# 编辑 conf/api_keys.env；该文件已被 Git 忽略。

python scripts/check_environment.py --strict
```

UniDepthV2 首次运行会从模型仓库下载约 1.5GiB 权重，需要可用网络或提前迁移 Hugging Face cache。MANO 文件受许可证限制，应从 MANO 官方渠道获取，不能随代码仓库发布。

## 5. 迁移验收

```bash
python -m compileall -q run_pipeline.py make_data.py hrse utils
python -m unittest discover -s tests -v
python run_pipeline.py --list-stages
python run_pipeline.py --video /path/input.mp4 --dry-run
```

最后用一段短视频逐阶段执行。先验和优化对 masks/规范帧很敏感，短视频 smoke test 也必须人工查看 overlay，不能只按进程退出码判断质量。
