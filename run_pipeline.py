#!/usr/bin/env python3
"""Run the complete ArtHOI video-to-4D-reconstruction pipeline.

The preprocessing stage remains interactive because SAM2 masks and the
canonical object/part separation must be inspected. Every expensive stage is
resume-friendly and writes a completion marker only after a successful exit.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from typing import Iterable


ROOT = Path(__file__).resolve().parent
STAGES = (
    "preprocess",
    "contact_windows",
    "contact_inference",
    "asr",
    "part_separation",
    "object_motion",
    "hoi_alignment",
)


def _load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE entries without executing shell syntax."""
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
        value = value.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if value:
            os.environ.setdefault(key, value)


def _default_sequence_name(video: Path) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", video.stem).strip("._-")
    return cleaned or "sequence"


def _validate_sequence_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise argparse.ArgumentTypeError(
            "sequence name must contain only letters, digits, '.', '_' and '-'"
        )
    return value


def _python_command(script: Path, args: Iterable[str], env_name: str = "current") -> list[str]:
    if env_name and env_name != "current":
        return [
            "conda",
            "run",
            "-n",
            env_name,
            "--live-stream",
            "python",
            str(script),
            *map(str, args),
        ]
    return [sys.executable, str(script), *map(str, args)]


def _require(paths: Iterable[Path], stage: str, dry_run: bool) -> None:
    if dry_run:
        return
    missing = [path for path in paths if not path.exists()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise RuntimeError(
            f"Stage '{stage}' is missing required inputs:\n{formatted}\n"
            "Resume the earlier stage or correct --from-stage."
        )


def _read_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("PyYAML is required to validate the sequence config") from error
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise RuntimeError(f"Config must contain a YAML mapping: {path}")
    return payload


def _validate_config(path: Path, part_count: int, canonical_frame: int) -> None:
    config = _read_yaml(path)
    actual_parts = int(config.get("part_cnt", -1))
    actual_frame = int(config.get("cano_frame", -1))
    if actual_parts != part_count or actual_frame != canonical_frame:
        raise RuntimeError(
            f"Config/video arguments disagree: {path} has part_cnt={actual_parts}, "
            f"cano_frame={actual_frame}; command requested {part_count}, {canonical_frame}."
        )
    ho_config = config.get("ho_align", {}) or {}
    selected = ho_config.get("xyz_pullback_instances")
    legacy_all_xyz = bool(ho_config.get("optimize_hand_xy", False))
    if selected is None and not legacy_all_xyz:
        raise RuntimeError(
            f"Left-hand XYZ pull-back is disabled in {path}. Set "
            "ho_align.xyz_pullback_instances: [left]."
        )
    if selected is not None:
        selected = [selected] if isinstance(selected, str) else list(selected)
        if "left" not in {str(item) for item in selected}:
            raise RuntimeError(
                f"Left-hand XYZ pull-back is not selected in {path}: {selected}"
            )


def _validate_left_contact(path: Path, allow_missing: bool) -> None:
    if allow_missing:
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    instances = payload.get("hand_instances")
    if isinstance(instances, dict):
        instance_ids = set(map(str, instances))
    else:
        instance_ids = set(map(str, payload.get("appeared", [])))
    if "left" not in instance_ids:
        raise RuntimeError(
            "The contact annotation does not contain a 'left' hand instance. "
            "Review processed/ho_contact.json or pass --allow-missing-left if intentional."
        )


class Pipeline:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.video = args.video.resolve()
        self.workspace = args.workspace.resolve()
        self.seq_name = args.seq_name or _default_sequence_name(self.video)
        self.seq_parent = self.workspace / "sequences"
        self.seq_path = self.seq_parent / self.seq_name
        self.config = (
            args.config.resolve()
            if args.config
            else self.workspace
            / "configs"
            / f"{self.seq_name}_p{args.part_cnt}c{args.cano_frame}.yml"
        )
        self.intermediate = self.workspace / "intermediate" / self.seq_name
        self.reconstruction = self.workspace / "reconstruction" / self.seq_name
        self.asr_parent = self.reconstruction / "asr"
        self.asr_path = self.asr_parent / self.seq_name
        self.parts_parent = self.reconstruction / "canonical_parts"
        self.parts_path = self.parts_parent / self.seq_name
        self.object_path = self.reconstruction / "object_4d"
        self.hoi_parent = self.reconstruction / "hoi"
        self.hoi_path = self.hoi_parent / self.seq_name
        self.contact_windows_parent = self.intermediate / "mllm_windows"
        self.contact_windows = self.contact_windows_parent / self.seq_name
        self.contact_responses = self.intermediate / "mllm_responses"
        self.contact_path = self.seq_path / "processed" / "ho_contact.json"
        self.marker_dir = self.workspace / ".pipeline" / self.seq_name
        self.env = dict(os.environ)
        self.env["ARTHOI_ENV"] = args.main_env
        self.env.setdefault("ARTHOI_DIFFUERASER_ENV", args.main_env)
        self.env.setdefault("ARTHOI_WILOR_ENV", args.main_env)
        self.env.setdefault("ARTHOI_PARTFIELD_ENV", args.main_env)

    def command(self, stage: str, command: list[str]) -> None:
        marker = self.marker_dir / f"{stage}.done"
        print(f"\n[{stage}] {shlex.join(command)}", flush=True)
        if self.args.dry_run:
            return
        if marker.is_file() and not self.args.force:
            print(f"[{stage}] completion marker found; skipping (use --force to rerun).")
            return
        subprocess.run(command, cwd=ROOT, env=self.env, check=True)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("completed\n", encoding="utf-8")

    def mark_internal_stage(self, stage: str) -> None:
        if self.args.dry_run:
            return
        marker = self.marker_dir / f"{stage}.done"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("completed\n", encoding="utf-8")

    def preprocess(self) -> None:
        command = _python_command(
            ROOT / "make_data.py",
            [
                "--video",
                self.video,
                "--seq-name",
                self.seq_name,
                "--out",
                self.seq_parent,
                "--part-cnt",
                self.args.part_cnt,
                "--cano-frame",
                self.args.cano_frame,
                "--target-reso",
                *self.args.target_reso,
                "--conf-out",
                self.config,
                *(["--skip-partfield"] if self.args.use_2d_mask else []),
            ],
        )
        self.command("preprocess", command)

    def contact_windows_stage(self) -> None:
        if self.args.contact_mode != "mllm":
            print(f"\n[contact_windows] skipped for contact mode '{self.args.contact_mode}'.")
            self.mark_internal_stage("contact_windows")
            return
        _require(
            [self.seq_path / "build" / "image", self.seq_path / "build" / "metric_depth.pkl"],
            "contact_windows",
            self.args.dry_run,
        )
        command = _python_command(
            ROOT / "hrse" / "mllm" / "seq_vlmho_gen.py",
            [
                "--seq-path",
                self.seq_path,
                "--output",
                self.contact_windows_parent,
                "--count",
                self.args.contact_window,
                "--interval",
                self.args.contact_interval,
                "--text-position",
                "top-right",
            ],
        )
        self.command("contact_windows", command)

    def contact_inference(self) -> None:
        if self.args.contact_mode == "mllm":
            _require([self.contact_windows], "contact_inference", self.args.dry_run)
            command = _python_command(
                ROOT / "hrse" / "mllm_contact.py",
                [
                    "--mllm-path",
                    self.contact_windows,
                    "--seq-name",
                    self.seq_name,
                    "--out",
                    self.seq_path / "processed",
                    "--response-dir",
                    self.contact_responses,
                    "--workers",
                    self.args.mllm_workers,
                    "--perspective",
                    self.args.perspective,
                    "--sequential-k",
                    self.args.contact_window,
                ],
            )
            self.command("contact_inference", command)
            return
        if self.args.contact_mode == "manual":
            _require([self.seq_path / "build" / "image"], "contact_inference", self.args.dry_run)
            command = _python_command(
                ROOT / "hrse" / "mllm" / "manual_label_ho.py",
                [
                    "--seq-path",
                    self.seq_path,
                    "--out",
                    "/processed/ho_contact.json",
                    "--mode",
                    "fingers",
                ],
            )
            self.command("contact_inference", command)
            return
        if not self.args.contact_json:
            raise RuntimeError("--contact-json is required with --contact-mode existing")
        source = self.args.contact_json.resolve()
        _require([source], "contact_inference", self.args.dry_run)
        print(f"\n[contact_inference] copy {source} -> {self.contact_path}")
        if not self.args.dry_run:
            self.contact_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, self.contact_path)
            self.mark_internal_stage("contact_inference")

    def asr(self) -> None:
        _require(
            [
                self.config,
                self.seq_path / "packed" / "visuals_inpainting.npz",
                self.seq_path / "build" / "mesh" / f"{self.args.cano_frame:05d}.obj",
            ],
            "asr",
            self.args.dry_run,
        )
        if self.config.is_file():
            _validate_config(self.config, self.args.part_cnt, self.args.cano_frame)
        command = _python_command(
            ROOT / "hrse" / "run_asr.py",
            [
                "--seq-path",
                self.seq_path,
                "--conf",
                self.config,
                "--out",
                self.asr_parent,
                "--seed",
                self.args.seed,
            ],
            self.args.asr_env,
        )
        self.command("asr", command)

    def part_separation(self) -> None:
        _require(
            [
                self.config,
                self.asr_path / "pred_pose.npy",
                self.asr_path / "final_mesh_asr.obj",
                self.seq_path / "processed" / "partseps",
            ],
            "part_separation",
            self.args.dry_run,
        )
        command_args = [
            "--seq-path",
            self.seq_path,
            "--conf",
            self.config,
            "--asr",
            self.asr_path,
            "--pfsep",
            self.seq_path / "processed" / "partseps",
            "--output-path",
            self.parts_parent,
        ]
        if self.args.use_2d_mask:
            command_args.append("--use_2d_mask")
        self.command(
            "part_separation",
            _python_command(ROOT / "hrse" / "object" / "part_seperation.py", command_args),
        )

    def object_motion(self) -> None:
        required = [self.config]
        required.extend(
            self.parts_path / f"part_{index}_partdata.npy"
            for index in range(self.args.part_cnt)
        )
        _require(required, "object_motion", self.args.dry_run)
        command = _python_command(
            ROOT / "hrse" / "object_4d.py",
            [
                "--seq-path",
                self.seq_path,
                "--conf",
                self.config,
                "--cano-reg-ckpt",
                self.parts_path,
                "--output-path",
                self.object_path,
            ],
        )
        self.command("object_motion", command)

    def hoi_alignment(self) -> None:
        required = [
            self.config,
            self.contact_path,
            self.seq_path / "processed" / "wilor_af" / "manoparam_fit.slerp.npy",
        ]
        for index in range(self.args.part_cnt):
            required.extend(
                [
                    self.object_path / f"part_{index}_partdata.npy",
                    self.object_path / f"part_{index}_best.npy",
                ]
            )
        _require(required, "hoi_alignment", self.args.dry_run)
        if self.config.is_file():
            _validate_config(self.config, self.args.part_cnt, self.args.cano_frame)
        if self.contact_path.is_file():
            _validate_left_contact(self.contact_path, self.args.allow_missing_left)
        command = _python_command(
            ROOT / "hrse" / "ho_align.py",
            [
                "--seq-path",
                self.seq_path,
                "--conf",
                self.config,
                "--fit-ckpt",
                self.object_path,
                "--output-path",
                self.hoi_parent,
                "--contact-path",
                self.contact_path,
                "--hand-prior",
                "wilor_af",
                "--initial-dump",
                "--dump-final",
            ],
        )
        self.command("hoi_alignment", command)

    def run(self) -> None:
        methods = {
            "preprocess": self.preprocess,
            "contact_windows": self.contact_windows_stage,
            "contact_inference": self.contact_inference,
            "asr": self.asr,
            "part_separation": self.part_separation,
            "object_motion": self.object_motion,
            "hoi_alignment": self.hoi_alignment,
        }
        start = STAGES.index(self.args.from_stage)
        end = STAGES.index(self.args.to_stage)
        if end < start:
            raise RuntimeError("--to-stage must not come before --from-stage")
        for stage in STAGES[start : end + 1]:
            methods[stage]()
        if self.args.dry_run:
            print("\nDry run complete; no pipeline commands were executed.")
        else:
            print("\nPipeline finished successfully.")
        print(f"Sequence workspace: {self.seq_path}")
        print(f"Object 4D reconstruction: {self.object_path}")
        print(f"Final hand-object reconstruction: {self.hoi_path}")
        print(f"Final videos: {self.hoi_path / 'final_opt'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, help="input RGB video")
    parser.add_argument("--seq-name", type=_validate_sequence_name, default=None)
    parser.add_argument("--workspace", type=Path, default=ROOT / "workspace")
    parser.add_argument("--config", type=Path, default=None, help="existing or generated sequence YAML")
    parser.add_argument("--part-cnt", type=int, default=2)
    parser.add_argument("--cano-frame", type=int, default=0)
    parser.add_argument("--target-reso", type=int, nargs=2, default=(960, 540), metavar=("W", "H"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--main-env", default=os.environ.get("ARTHOI_ENV", "arthoi"))
    parser.add_argument(
        "--asr-env",
        default=os.environ.get(
            "ARTHOI_ASR_ENV", os.environ.get("ARTHOI_ENV", "arthoi")
        ),
    )
    parser.add_argument("--env-file", type=Path, default=ROOT / "conf" / "api_keys.env")
    parser.add_argument("--contact-mode", choices=("mllm", "manual", "existing"), default="mllm")
    parser.add_argument("--contact-json", type=Path, default=None)
    parser.add_argument("--contact-window", type=int, default=3)
    parser.add_argument("--contact-interval", type=int, default=1)
    parser.add_argument("--mllm-workers", type=int, default=1)
    parser.add_argument("--perspective", choices=("auto", "1", "3"), default="auto")
    parser.add_argument("--use-2d-mask", action="store_true", help="force 2D-mask mesh separation")
    parser.add_argument("--allow-missing-left", action="store_true")
    parser.add_argument("--from-stage", choices=STAGES, default=STAGES[0])
    parser.add_argument("--to-stage", choices=STAGES, default=STAGES[-1])
    parser.add_argument("--force", action="store_true", help="rerun stages even when completion markers exist")
    parser.add_argument("--dry-run", action="store_true", help="print all selected commands without running them")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--doctor", action="store_true", help="check the installation and exit")
    parser.add_argument("--list-stages", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.list_stages:
        print("\n".join(STAGES))
        return 0
    _load_env_file(args.env_file.resolve())
    if args.doctor:
        command = [
            sys.executable,
            str(ROOT / "scripts" / "check_environment.py"),
            "--contact-mode",
            args.contact_mode,
        ]
        if args.use_2d_mask:
            command.append("--skip-partfield")
        return subprocess.run(command, cwd=ROOT).returncode
    if args.video is None:
        parser.error("--video is required unless --doctor or --list-stages is used")
    if not args.video.is_file():
        parser.error(f"input video does not exist: {args.video}")
    if args.part_cnt < 1:
        parser.error("--part-cnt must be at least 1")
    if args.cano_frame < 0:
        parser.error("--cano-frame must be non-negative")
    if min(args.target_reso) <= 0:
        parser.error("--target-reso dimensions must be positive")
    if args.contact_mode == "existing" and args.contact_json is None:
        parser.error("--contact-json is required with --contact-mode existing")
    if not args.skip_preflight and not args.dry_run:
        doctor = [
            sys.executable,
            str(ROOT / "scripts" / "check_environment.py"),
            "--strict",
            "--contact-mode",
            args.contact_mode,
        ]
        if args.use_2d_mask:
            doctor.append("--skip-partfield")
        subprocess.run(doctor, cwd=ROOT, check=True)
    Pipeline(args).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
