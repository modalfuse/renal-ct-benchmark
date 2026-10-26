#!/usr/bin/env python3
"""FMCIB single-case check with official get_features and the Zenodo weights.

Set L4_ROOT and L4_ROI. The example case id is 08FBroxzI6.
A failed check stops the process and does not start the full run.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(os.environ["L4_ROOT"])
WEIGHT = ROOT / "weights" / "fmcib" / "model_weights.torch"
PID = "08FBroxzI6"
ROI = Path(os.environ["L4_ROI"])
IMG = ROI / f"{PID}_img.nii.gz"
MASK = ROI / f"{PID}_mask.nii.gz"
CSV_PATH = ROOT / "data" / "fmcib_singleton.csv"


def centroid_lps(img_path: Path, mask_path: Path):
    image = nib.load(str(img_path))
    mask = np.asanyarray(nib.load(str(mask_path)).dataobj) > 0
    if image.shape[:3] != mask.shape[:3]:
        raise RuntimeError(f"grid_mismatch img {image.shape} mask {mask.shape}")
    pts = np.argwhere(mask)
    if pts.size == 0:
        raise RuntimeError("empty_mask")
    center = pts.mean(axis=0)
    ras = image.affine @ np.array([center[0], center[1], center[2], 1.0])
    return float(-ras[0]), float(-ras[1]), float(ras[2])


def main():
    from l4_probe import refuse_disallowed
    refuse_disallowed([IMG, MASK, WEIGHT])
    if not WEIGHT.is_file() or WEIGHT.stat().st_size < 1000000:
        raise SystemExit(f"weight missing {WEIGHT}")
    if not IMG.is_file() or not MASK.is_file():
        raise SystemExit("missing roi")
    x, y, z = centroid_lps(IMG, MASK)
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "coordX", "coordY", "coordZ"])
        writer.writeheader()
        writer.writerow({
            "image_path": str(IMG),
            "coordX": f"{x:.6f}",
            "coordY": f"{y:.6f}",
            "coordZ": f"{z:.6f}",
        })
    print(f"centroid_lps {x:.3f} {y:.3f} {z:.3f}", flush=True)
    sys.path.insert(0, str(ROOT / "repos" / "FMCIB"))
    from fmcib.run import get_features

    frame = get_features(
        str(CSV_PATH),
        weights_path=str(WEIGHT),
        batch_size=1,
        num_workers=0,
    )
    cols = [c for c in frame.columns if c.startswith("pred_")]
    if len(cols) != 4096:
        raise SystemExit(f"dim {len(cols)}")
    vector = frame.loc[0, cols].to_numpy(dtype=float)
    if not np.isfinite(vector).all():
        raise SystemExit("non_finite")
    out = ROOT / "outputs" / "fmcib"
    out.mkdir(parents=True, exist_ok=True)
    payload = {"pid": PID, "dim": 4096, "finite": True, "weight_bytes": WEIGHT.stat().st_size}
    (out / "singleton.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")
    print("SINGLETON_OK", json.dumps(payload), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        err = ROOT / "outputs" / "fmcib" / "feature_errors_fmcib.csv"
        err.parent.mkdir(parents=True, exist_ok=True)
        err.write_text(f"pid,cohort,reason,detail\n{PID},ucsf,extract_failed,{type(exc).__name__}: {exc}\n", encoding="utf-8")
        print("SINGLETON_FAIL", exc, flush=True)
        raise
