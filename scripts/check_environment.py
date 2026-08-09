#!/usr/bin/env python3
"""Check ArtHOI source layout, runtime dependencies, weights and credentials."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]

CORE_FILES = (
    "make_data.py",
    "run_pipeline.py",
    "conf/template.yml",
    "hrse/run_asr.py",
    "hrse/object/part_seperation.py",
    "hrse/object_4d.py",
    "hrse/ho_align.py",
    "hrse/hand_instances.py",
    "hrse/mllm_contact.py",
    "third_party/sam2/app_hrse.py",
    "third_party/DiffuEraser/run_diffueraser.py",
    "third_party/Video-Depth-Anything/run.py",
    "third_party/unidepth/metric_align.py",
    "third_party/WiLoR/WiLoR_ArtHOI.py",
    "third_party/PartField/partfield_inference.py",
    "third_party/PartField/partfield/run_seperate.py",
)

PYTHON_MODULES = (
    "torch",
    "torchvision",
    "numpy",
    "cv2",
    "PIL",
    "yaml",
    "omegaconf",
    "trimesh",
    "open3d",
    "smplx",
    "nvdiffrast",
    "pytorch3d",
    "manopth",
    "cotracker",
    "unidepth",
    "lightning",
    "gradio",
)

REQUIRED_FILES = (
    ("SAM2", "third_party/sam2/checkpoints/sam2.1_hiera_base_plus.pt"),
    ("Video-Depth-Anything", "third_party/Video-Depth-Anything/checkpoints/video_depth_anything_vits.pth"),
    ("CoTracker online", "third_party/co-tracker/checkpoints/scaled_online.pth"),
    ("FoundationPose scorer", "third_party/foundationpose/weights/2024-01-11-20-02-45/model_best.pth"),
    ("FoundationPose scorer config", "third_party/foundationpose/weights/2024-01-11-20-02-45/config.yml"),
    ("FoundationPose refiner", "third_party/foundationpose/weights/2023-10-28-18-33-37/model_best.pth"),
    ("FoundationPose refiner config", "third_party/foundationpose/weights/2023-10-28-18-33-37/config.yml"),
    ("WiLoR", "third_party/WiLoR/pretrained_models/wilor_final.ckpt"),
    ("WiLoR detector", "third_party/WiLoR/pretrained_models/detector.pt"),
    ("WiLoR MANO", "third_party/WiLoR/mano_data/MANO_RIGHT.pkl"),
    ("MANO left", "third_party/body_models/MANO_LEFT.pkl"),
    ("MANO right", "third_party/body_models/MANO_RIGHT.pkl"),
)

REQUIRED_DIRECTORIES = (
    ("DiffuEraser Stable Diffusion", "third_party/DiffuEraser/weights/stable-diffusion-v1-5"),
    ("DiffuEraser VAE", "third_party/DiffuEraser/weights/sd-vae-ft-mse"),
    ("DiffuEraser model", "third_party/DiffuEraser/weights/diffuEraser"),
    ("ProPainter", "third_party/DiffuEraser/weights/propainter"),
)


def _load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and value:
            os.environ.setdefault(key, value)


def _usable_secret(name: str) -> bool:
    value = os.environ.get(name, "").strip()
    if not value:
        return False
    upper = value.upper()
    return not upper.startswith(("REPLACE_", "CHANGE_ME", "INVALID"))


def _directory_has_payload(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(item.is_file() and item.name.lower() != "readme.md" for item in path.rglob("*"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="fail on missing dependencies or weights")
    parser.add_argument("--contact-mode", choices=("mllm", "manual", "existing"), default="mllm")
    parser.add_argument("--skip-partfield", action="store_true")
    args = parser.parse_args()
    _load_env_file(ROOT / "conf" / "api_keys.env")

    fatal: list[str] = []
    missing: list[str] = []
    warnings: list[str] = []

    print(f"ArtHOI root: {ROOT}")
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")

    for relative in CORE_FILES:
        path = ROOT / relative
        if not path.is_file():
            fatal.append(f"core source: {relative}")

    for executable in ("ffmpeg", "ffprobe", "conda", "nvidia-smi"):
        if shutil.which(executable) is None:
            missing.append(f"executable: {executable}")

    for module in PYTHON_MODULES:
        if args.skip_partfield and module == "lightning":
            continue
        try:
            found = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            missing.append(f"python module: {module}")

    for label, relative in REQUIRED_FILES:
        if not (ROOT / relative).is_file():
            missing.append(f"weight [{label}]: {relative}")

    for label, relative in REQUIRED_DIRECTORIES:
        if not _directory_has_payload(ROOT / relative):
            missing.append(f"weight directory [{label}]: {relative}")

    if not args.skip_partfield:
        partfield = ROOT / "third_party/PartField/model/model_objaverse.ckpt"
        if not partfield.is_file():
            missing.append(
                "weight [PartField]: third_party/PartField/model/model_objaverse.ckpt"
            )

    native_candidates = list(
        (ROOT / "third_party/foundationpose/mycpp").glob("build/**/mycpp*.so")
    )
    if not native_candidates:
        missing.append("native extension: FoundationPose mycpp (run its build_all_conda.sh)")

    if args.contact_mode == "mllm" and not (
        _usable_secret("QNAIGC_API_KEY") or _usable_secret("QWEN_API_KEY")
    ):
        missing.append("MLLM credential: set QNAIGC_API_KEY or QWEN_API_KEY")

    if not (
        _usable_secret("TENCENTMAAS_API_KEY")
        or (_usable_secret("TENCENTCLOUD_SECRET_ID") and _usable_secret("TENCENTCLOUD_SECRET_KEY"))
    ):
        warnings.append(
            "No Hunyuan3D credential detected; place a canonical GLB manually when prompted."
        )
    warnings.append(
        "UniDepthV2 downloads lpiccinelli/unidepth-v2-vitl14 through the model cache on first use."
    )

    if not fatal and not missing:
        print("\nOK: required source, dependencies and local weights were found.")
    for item in fatal:
        print(f"FATAL: {item}")
    for item in missing:
        print(f"MISSING: {item}")
    for item in warnings:
        print(f"NOTE: {item}")

    print(
        f"\nSummary: fatal={len(fatal)} missing={len(missing)} "
        f"notes={len(warnings)} strict={args.strict}"
    )
    return 1 if fatal or (args.strict and missing) else 0


if __name__ == "__main__":
    raise SystemExit(main())
