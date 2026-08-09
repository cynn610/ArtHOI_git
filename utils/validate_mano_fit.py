"""Validate that serialized MANO parameters reproduce WiLoR hand meshes."""

import argparse
from pathlib import Path

import numpy as np
import smplx
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-path", required=True)
    parser.add_argument("--model-dir", default="wilor_af")
    parser.add_argument("--max-rmse", type=float, default=0.08)
    args = parser.parse_args()

    seq_path = Path(args.seq_path)
    model_root = Path(__file__).resolve().parents[1] / "third_party" / "body_models"
    params = np.load(
        seq_path / "processed" / args.model_dir / "manoparam_fit.slerp.npy",
        allow_pickle=True,
    ).item()
    targets = np.load(
        seq_path / "processed" / args.model_dir / "v3d.npy",
        allow_pickle=True,
    ).item()

    failures = []
    for side in ("left", "right"):
        model = smplx.create(
            str(model_root / f"MANO_{side.upper()}.pkl"),
            model_type="mano",
            use_pca=False,
            is_rhand=(side == "right"),
        )
        data = params[side]
        target = np.asarray(targets[f"v3d.{side}"])
        valid = np.isfinite(target).all(axis=(1, 2))
        valid &= np.isfinite(np.asarray(data["betas"])).all(axis=1)
        valid &= np.isfinite(np.asarray(data["global_orient"])).all(axis=1)
        valid &= np.isfinite(np.asarray(data["hand_pose"])).all(axis=1)
        valid &= np.isfinite(np.asarray(data["transl"])).all(axis=1)
        if not valid.any():
            print(f"{side}: no valid frames")
            continue

        with torch.no_grad():
            fitted = model(
                betas=torch.from_numpy(np.asarray(data["betas"], dtype=np.float32)),
                global_orient=torch.from_numpy(
                    np.asarray(data["global_orient"], dtype=np.float32)
                ),
                hand_pose=torch.from_numpy(
                    np.asarray(data["hand_pose"], dtype=np.float32)
                ),
                transl=torch.from_numpy(np.asarray(data["transl"], dtype=np.float32)),
            ).vertices.numpy()

        rmse = np.sqrt(np.mean((fitted[valid] - target[valid]) ** 2, axis=(1, 2)))
        centroid = np.linalg.norm(
            fitted[valid].mean(axis=1) - target[valid].mean(axis=1), axis=1
        )
        print(
            f"{side}: frames={valid.sum()} "
            f"vertex_rmse_median={np.median(rmse):.4f}m "
            f"vertex_rmse_max={np.max(rmse):.4f}m "
            f"centroid_error_median={np.median(centroid):.4f}m"
        )
        if float(np.median(rmse)) > args.max_rmse:
            failures.append(side)

    if failures:
        raise SystemExit(
            "MANO registration validation failed for: " + ", ".join(failures)
        )


if __name__ == "__main__":
    main()
