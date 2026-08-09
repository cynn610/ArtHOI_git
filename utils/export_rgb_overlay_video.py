"""Overlay rendered reconstruction panels on the original RGB frames."""

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


def collect_frames(directory: Path, pattern: str) -> list[Path]:
    frames = sorted(directory.glob(pattern), key=frame_number)
    if not frames:
        raise FileNotFoundError(f"No frames matching {pattern!r} in {directory}")
    return frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay ArtHOI rendered meshes on clean RGB input frames."
    )
    parser.add_argument("rgb_dir", type=Path, help="Directory of clean RGB frames")
    parser.add_argument(
        "visualization_dir", type=Path, help="Directory of pred_*.png RGB grids"
    )
    parser.add_argument("output", type=Path, help="Output .mp4 path")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Rendered mesh opacity in [0, 1] (default: 0.7)",
    )
    parser.add_argument(
        "--match-by-order",
        action="store_true",
        help=(
            "Pair sorted RGB and visualization frames by position instead of "
            "requiring identical numeric suffixes"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("FPS must be greater than zero")
    if not 0 <= args.alpha <= 1:
        raise ValueError("Alpha must be between 0 and 1")

    rgb_frames = collect_frames(args.rgb_dir, "*.png")
    vis_frames = collect_frames(args.visualization_dir, "pred_*.png")
    rgb_numbers = [frame_number(path) for path in rgb_frames]
    vis_numbers = [frame_number(path) for path in vis_frames]
    if len(rgb_frames) != len(vis_frames):
        raise ValueError(
            "RGB and visualization frame counts do not match: "
            f"{len(rgb_frames)} != {len(vis_frames)}"
        )
    if not args.match_by_order and rgb_numbers != vis_numbers:
        raise ValueError("RGB and visualization frame indices do not match")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    size = None
    with imageio.get_writer(
        args.output,
        format="FFMPEG",
        mode="I",
        fps=args.fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=1,
        output_params=["-crf", "18", "-movflags", "+faststart"],
    ) as writer:
        for rgb_path, vis_path in zip(rgb_frames, vis_frames):
            with Image.open(rgb_path) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
            with Image.open(vis_path) as image:
                visualization = np.asarray(image.convert("RGB"))

            height, width = rgb.shape[:2]
            if visualization.shape[0] < height or visualization.shape[1] < 2 * width:
                raise ValueError(
                    f"Visualization grid {vis_path} is too small for RGB size "
                    f"{width}x{height}"
                )
            if size is None:
                size = (width, height)
                if width % 2 or height % 2:
                    raise ValueError(f"RGB dimensions must be even, got {size}")
            elif (width, height) != size:
                raise ValueError(f"Inconsistent RGB frame size at {rgb_path}")

            rendered = visualization[:height, :width].astype(np.float32)
            rendered_mask = (
                visualization[:height, width : 2 * width, 0].astype(np.float32)
                / 255.0
            )
            blend = (args.alpha * rendered_mask)[..., None]
            overlay = rgb * (1.0 - blend) + rendered * blend
            writer.append_data(np.clip(overlay, 0, 255).astype(np.uint8))

    assert size is not None
    duration = len(rgb_frames) / args.fps
    print(
        f"Wrote {args.output} ({len(rgb_frames)} frames, {size[0]}x{size[1]}, "
        f"{args.fps:g} FPS, {duration:.2f}s)"
    )


if __name__ == "__main__":
    main()
