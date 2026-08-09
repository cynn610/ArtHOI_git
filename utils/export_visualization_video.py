"""Export numerically named visualization frames as an MP4 video."""

import argparse
import re
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode a directory of visualization frames as H.264 MP4."
    )
    parser.add_argument("frames_dir", type=Path, help="Directory containing frames")
    parser.add_argument("output", type=Path, help="Output .mp4 path")
    parser.add_argument(
        "--pattern", default="pred_*.png", help="Frame glob (default: pred_*.png)"
    )
    parser.add_argument("--fps", type=float, default=10.0, help="Output FPS")
    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="H.264 quality: lower is better (default: 18)",
    )
    return parser.parse_args()


def frame_number(path: Path) -> int:
    match = re.search(r"(\d+)(?=\.[^.]+$)", path.name)
    if match is None:
        raise ValueError(f"Frame name has no numeric suffix: {path.name}")
    return int(match.group(1))


def collect_frames(frames_dir: Path, pattern: str) -> list[Path]:
    frames = sorted(frames_dir.glob(pattern), key=frame_number)
    if not frames:
        raise FileNotFoundError(f"No frames matching {pattern!r} in {frames_dir}")

    numbers = [frame_number(frame) for frame in frames]
    expected = list(range(numbers[0], numbers[-1] + 1))
    if numbers != expected:
        missing = sorted(set(expected) - set(numbers))
        raise ValueError(f"Frame sequence is not contiguous; missing: {missing}")
    return frames


def export_video(
    frames: list[Path], output: Path, fps: float, crf: int
) -> tuple[int, int]:
    if fps <= 0:
        raise ValueError("FPS must be greater than zero")
    if not 0 <= crf <= 51:
        raise ValueError("CRF must be between 0 and 51")

    output.parent.mkdir(parents=True, exist_ok=True)
    expected_size = None
    with imageio.get_writer(
        output,
        format="FFMPEG",
        mode="I",
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=1,
        output_params=["-crf", str(crf), "-movflags", "+faststart"],
    ) as writer:
        for frame_path in frames:
            with Image.open(frame_path) as image:
                rgb = np.asarray(image.convert("RGB"))
            size = (rgb.shape[1], rgb.shape[0])
            if expected_size is None:
                expected_size = size
                if size[0] % 2 or size[1] % 2:
                    raise ValueError(
                        f"Frame dimensions must be even for yuv420p, got {size}"
                    )
            elif size != expected_size:
                raise ValueError(
                    f"Inconsistent frame size: {frame_path} is {size}, "
                    f"expected {expected_size}"
                )
            writer.append_data(rgb)

    assert expected_size is not None
    return expected_size


def main() -> None:
    args = parse_args()
    frames = collect_frames(args.frames_dir, args.pattern)
    width, height = export_video(frames, args.output, args.fps, args.crf)
    duration = len(frames) / args.fps
    print(
        f"Wrote {args.output} ({len(frames)} frames, {width}x{height}, "
        f"{args.fps:g} FPS, {duration:.2f}s)"
    )


if __name__ == "__main__":
    main()
