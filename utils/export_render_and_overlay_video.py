"""Export Predicted Rendered and RGB Overlapped panels side by side."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image


def frame_number(path: Path) -> int:
    match = re.search(r"(\d+)(?=\.[^.]+$)", path.name)
    if match is None:
        raise ValueError(f"Frame name has no numeric suffix: {path.name}")
    return int(match.group(1))


def collect_frames(frames_dir: Path) -> list[Path]:
    frames = sorted(frames_dir.glob("pred_*.png"), key=frame_number)
    if not frames:
        raise FileNotFoundError(f"No pred_*.png frames in {frames_dir}")
    numbers = [frame_number(frame) for frame in frames]
    expected = list(range(numbers[0], numbers[-1] + 1))
    if numbers != expected:
        missing = sorted(set(expected).difference(numbers))
        raise ValueError(f"Frame sequence is not contiguous; missing: {missing}")
    return frames


def extract_panels(frame: np.ndarray) -> np.ndarray:
    """Extract top-left rendered and bottom-left RGB-overlaid grid panels."""
    grid_height, grid_width = frame.shape[:2]
    if grid_width % 2 or grid_height % 3:
        raise ValueError(
            "ArtHOI visualization grid must be 2 columns by 3 rows, got "
            f"{grid_width}x{grid_height}"
        )
    panel_width = grid_width // 2
    panel_height = grid_height // 3
    predicted_rendered = frame[:panel_height, :panel_width]
    rgb_overlapped = frame[2 * panel_height :, :panel_width]
    return np.concatenate((predicted_rendered, rgb_overlapped), axis=1)


def export_video(
    frames: list[Path], output: Path, fps: float = 30.0, crf: int = 18
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
                frame = np.asarray(image.convert("RGB"))
            combined = extract_panels(frame)
            size = (combined.shape[1], combined.shape[0])
            if expected_size is None:
                expected_size = size
                if size[0] % 2 or size[1] % 2:
                    raise ValueError(f"Output dimensions must be even, got {size}")
            elif size != expected_size:
                raise ValueError(
                    f"Inconsistent frame size at {frame_path}: {size} != {expected_size}"
                )
            writer.append_data(combined)

    assert expected_size is not None
    return expected_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--crf", type=int, default=18)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = collect_frames(args.frames_dir)
    width, height = export_video(frames, args.output, args.fps, args.crf)
    print(
        f"Wrote {args.output} ({len(frames)} frames, {width}x{height}, "
        f"{args.fps:g} FPS, {len(frames) / args.fps:.2f}s)"
    )


if __name__ == "__main__":
    main()
