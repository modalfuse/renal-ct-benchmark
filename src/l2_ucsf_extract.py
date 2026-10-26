#!/usr/bin/env python3
"""Read-only UCSF radiomics extraction for the L2 baseline.

The classification reader loads roi/<pid>_img.nii.gz with roi/<pid>_mask.nii.gz.
That pair shares one grid. The full hdf5 portven volume does not. This script
checks that each hdf5 file exists and that portven's shape matches qc_seg, then
extracts features from the ROI image/mask pair. Spacing is taken from qc_seg.
"""

from __future__ import annotations

import argparse
import logging
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_CLASSES = ("shape", "firstorder", "glcm", "glrlm", "glszm", "ngtdm", "gldm")


def parse_args():
    p = argparse.ArgumentParser(description="Extract UCSF PV radiomics from ROI crops")
    p.add_argument("--hdf5-dir", type=Path, default=None)
    p.add_argument("--roi-dir", type=Path, default=None)
    p.add_argument("--qc-seg", type=Path, default=None)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--errors", type=Path, default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--tasks", type=Path, default=None, help="external pid,cohort,label,phase,img,mask CSV")
    return p.parse_args()


def parse_spacing(text: str):
    parts = [float(x) for x in str(text).split("x")]
    if len(parts) != 3 or any(v <= 0 for v in parts):
        raise ValueError(f"bad spacing {text}")
    return tuple(parts)


def make_extractor():
    import radiomics
    from radiomics import featureextractor

    radiomics.setVerbosity(logging.ERROR)
    ext = featureextractor.RadiomicsFeatureExtractor(
        binWidth=25,
        label=1,
        geometryTolerance=1e-5,
        force2D=False,
        additionalInfo=True,
    )
    ext.disableAllFeatures()
    for name in FEATURE_CLASSES:
        ext.enableFeatureClassByName(name)
    ext.disableAllImageTypes()
    ext.enableImageTypeByName("Original")
    ext.settings["normalize"] = False
    ext.settings["resampledPixelSpacing"] = None
    return ext


def sitk_pair(image, mask, spacing):
    import SimpleITK as sitk

    img = sitk.GetImageFromArray(np.transpose(np.asarray(image, dtype=np.float32), (2, 1, 0)))
    msk = sitk.GetImageFromArray(np.transpose(np.asarray(mask, dtype=np.uint8), (2, 1, 0)))
    for vol in (img, msk):
        vol.SetSpacing(tuple(float(v) for v in spacing))
        vol.SetOrigin((0.0, 0.0, 0.0))
        vol.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    return img, msk


def mask_ok(mask: np.ndarray):
    if mask.dtype == np.bool_:
        return True, mask.astype(np.uint8)
    vals = set(np.unique(mask).tolist())
    if vals <= {0, 1}:
        return True, mask.astype(np.uint8)
    return False, None


def one_case(task):
    import nibabel as nib

    pid = task["pid"]
    errors = []
    try:
        import h5py

        h5_path = Path(task["h5"])
        if not h5_path.is_file():
            return {"pid": pid, "errors": [{"reason": "hdf5_missing", "detail": str(h5_path)}]}
        with h5py.File(h5_path, "r") as handle:
            if "portven" not in handle:
                return {"pid": pid, "errors": [{"reason": "hdf5_missing", "detail": "portven"}]}
            vol_shape = tuple(int(v) for v in handle["portven"].shape)
        qc_shape = tuple(int(v) for v in task["qc_shape"].split("x"))
        if vol_shape != qc_shape:
            return {
                "pid": pid,
                "errors": [{"reason": "grid_mismatch", "detail": f"portven {vol_shape} qc {qc_shape}"}],
            }
        img_path = Path(task["img"])
        msk_path = Path(task["mask"])
        if not img_path.is_file() or not msk_path.is_file():
            return {"pid": pid, "errors": [{"reason": "missing_file", "detail": "roi image or mask"}]}
        image = np.asanyarray(nib.load(str(img_path)).dataobj)
        mask = np.asanyarray(nib.load(str(msk_path)).dataobj)
        if image.shape != mask.shape:
            return {
                "pid": pid,
                "errors": [{"reason": "grid_mismatch", "detail": f"img {image.shape} mask {mask.shape}"}],
            }
        ok, mask_u8 = mask_ok(mask)
        if not ok:
            return {"pid": pid, "errors": [{"reason": "mask_values", "detail": str(np.unique(mask)[:8])}]}
        if int((mask_u8 == 1).sum()) == 0:
            return {"pid": pid, "errors": [{"reason": "empty_mask", "detail": "no label 1"}]}
        vmin = float(np.min(image))
        vmax = float(np.max(image))
        if not (vmin < -100 and vmax > 200):
            errors.append({"reason": "hu_range", "detail": f"min={vmin} max={vmax}"})
        spacing = parse_spacing(task["spacing"])
        sitk_img, sitk_msk = sitk_pair(image, mask_u8, spacing)
        feats = make_extractor().execute(sitk_img, sitk_msk)
        row = {
            "pid": pid,
            "spacing_x": spacing[0],
            "spacing_y": spacing[1],
            "spacing_z": spacing[2],
        }
        for key, value in feats.items():
            if not str(key).startswith("original_"):
                continue
            try:
                num = float(value)
            except (TypeError, ValueError):
                num = float("nan")
            row[key] = num
        return {"pid": pid, "row": row, "errors": errors}
    except Exception as exc:
        errors.append({"reason": "extract_failed", "detail": f"{type(exc).__name__}: {exc}"})
        return {"pid": pid, "errors": errors}


def ext_case(task):
    import nibabel as nib

    pid = task["pid"]
    errors = []
    try:
        img_path = Path(task["img"])
        msk_path = Path(task["mask"])
        if not img_path.is_file() or not msk_path.is_file():
            return {"pid": pid, "errors": [{"reason": "missing_file", "detail": "roi image or mask"}]}
        img_nii = nib.load(str(img_path))
        msk_nii = nib.load(str(msk_path))
        image = np.asanyarray(img_nii.dataobj)
        mask = np.asanyarray(msk_nii.dataobj)
        z_img = tuple(float(v) for v in img_nii.header.get_zooms()[:3])
        z_msk = tuple(float(v) for v in msk_nii.header.get_zooms()[:3])
        if image.shape != mask.shape or any(abs(a - b) > 1e-5 for a, b in zip(z_img, z_msk)):
            return {
                "pid": pid,
                "errors": [{"reason": "grid_mismatch", "detail": f"img {image.shape} mask {mask.shape}"}],
            }
        ok, mask_u8 = mask_ok(mask)
        if not ok:
            return {"pid": pid, "errors": [{"reason": "mask_values", "detail": str(np.unique(mask)[:8])}]}
        if int((mask_u8 == 1).sum()) == 0:
            return {"pid": pid, "errors": [{"reason": "empty_mask", "detail": "no label 1"}]}
        vmin = float(np.min(image))
        vmax = float(np.max(image))
        if not (vmin < -100 and vmax > 200):
            errors.append({"reason": "hu_range", "detail": f"min={vmin} max={vmax}"})
        sitk_img, sitk_msk = sitk_pair(image, mask_u8, z_img)
        feats = make_extractor().execute(sitk_img, sitk_msk)
        row = {
            "pid": pid,
            "cohort": task["cohort"],
            "label": task["label"],
            "phase": task["phase"],
            "spacing_x": z_img[0],
            "spacing_y": z_img[1],
            "spacing_z": z_img[2],
        }
        for key, value in feats.items():
            if not str(key).startswith("original_"):
                continue
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                row[key] = float("nan")
        return {"pid": pid, "row": row, "errors": errors, "cohort": task["cohort"]}
    except Exception as exc:
        errors.append({"reason": "extract_failed", "detail": f"{type(exc).__name__}: {exc}"})
        return {"pid": pid, "errors": errors, "cohort": task["cohort"]}


def main():
    args = parse_args()
    import radiomics
    import SimpleITK as sitk

    if args.tasks is not None:
        tasks_df = pd.read_csv(args.tasks, dtype=str)
        tasks = tasks_df.to_dict("records")
        print(f"n_tasks={len(tasks)}", flush=True)
        print(f"pyradiomics={radiomics.__version__}", flush=True)
        print(f"simpleitk={sitk.Version_VersionString()}", flush=True)
        ctx = get_context("fork")
        with ctx.Pool(args.workers) as pool:
            results = list(pool.imap_unordered(ext_case, tasks, chunksize=1))
        rows, errors = [], []
        for item in results:
            for err in item.get("errors") or []:
                errors.append({"pid": item["pid"], "cohort": item.get("cohort", ""), **err})
            if "row" in item:
                item["row"]["pyradiomics_version"] = radiomics.__version__
                item["row"]["simpleitk_version"] = sitk.Version_VersionString()
                rows.append(item["row"])
        frame = pd.DataFrame(rows)
        err_df = pd.DataFrame(errors, columns=["pid", "cohort", "reason", "detail"])
        args.out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.out, index=False)
        err_path = args.errors or args.out.with_name("ext_feature_errors.csv")
        err_df.to_csv(err_path, index=False)
        n_feat = sum(c.startswith("original_") for c in frame.columns) if len(frame) else 0
        print(f"n_ok={len(frame)} n_feature_cols={n_feat} n_error_rows={len(err_df)}", flush=True)
        hard = err_df[err_df["reason"] != "hu_range"] if len(err_df) else err_df
        if len(frame) != len(tasks) or len(hard):
            raise SystemExit(f"incomplete extraction ok={len(frame)} expected={len(tasks)} hard_errors={len(hard)}")
        return

    if args.hdf5_dir is None or args.roi_dir is None or args.qc_seg is None:
        raise SystemExit("UCSF extraction requires --hdf5-dir, --roi-dir and --qc-seg")
    qc = pd.read_csv(args.qc_seg, dtype={"pid": str})
    qc = qc[qc["status"].astype(str) == "ok"].copy()
    tasks = []
    for rec in qc.itertuples(index=False):
        pid = str(rec.pid)
        tasks.append(
            {
                "pid": pid,
                "h5": str(args.hdf5_dir / f"{pid}.hdf5"),
                "img": str(args.roi_dir / f"{pid}_img.nii.gz"),
                "mask": str(args.roi_dir / f"{pid}_mask.nii.gz"),
                "spacing": rec.spacing,
                "qc_shape": rec.shape,
            }
        )
    print(f"n_qc={len(tasks)}", flush=True)
    print(f"pyradiomics={radiomics.__version__}", flush=True)
    print(f"simpleitk={sitk.Version_VersionString()}", flush=True)
    ctx = get_context("fork")
    with ctx.Pool(args.workers) as pool:
        results = list(pool.imap_unordered(one_case, tasks, chunksize=1))
    rows = []
    errors = []
    for item in results:
        for err in item.get("errors") or []:
            errors.append({"pid": item["pid"], "cohort": "ucsf", **err})
        if "row" in item:
            item["row"]["pyradiomics_version"] = radiomics.__version__
            item["row"]["simpleitk_version"] = sitk.Version_VersionString()
            rows.append(item["row"])
    frame = pd.DataFrame(rows).sort_values("pid")
    err_df = pd.DataFrame(errors, columns=["pid", "cohort", "reason", "detail"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    err_path = args.errors or args.out.with_name("ucsf_feature_errors.csv")
    err_df.to_csv(err_path, index=False)
    n_feat = sum(c.startswith("original_") for c in frame.columns) if len(frame) else 0
    print(f"n_ok={len(frame)} n_feature_cols={n_feat} n_error_rows={len(err_df)}", flush=True)
    print("wrote", args.out, flush=True)
    print("wrote", err_path, flush=True)
    hard = err_df[err_df["reason"] != "hu_range"] if len(err_df) else err_df
    if len(frame) != len(tasks) or len(hard):
        raise SystemExit(f"incomplete extraction ok={len(frame)} expected={len(tasks)} hard_errors={len(hard)}")


if __name__ == "__main__":
    main()
