#!/usr/bin/env python3
"""FMCIB full run: official 1 mm, 50x50x50 lesion patches, 4096-d features, then linear probing.

Set L4_ROOT to the working directory that contains repos/, weights/, features/, and outputs/.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from l4_probe import refuse_disallowed, run_probe
from l4_store import ErrorLog, FeatureStore
from l4_tables import read_cases

ROOT = Path(os.environ["L4_ROOT"])
REPO = ROOT / "repos" / "FMCIB"
WEIGHT = ROOT / "weights" / "fmcib" / "model_weights.torch"
EXPECT_COMMIT = "1f1c0c8725c110c9c70cb466467a55e3160760c9"
EXPECT_MD5 = "b645e4d99c12adbabd198c12f22efaa4"
EXPECT_SIZE = 738451713
PID = "08FBroxzI6"
FM = "fmcib"


def read_commit(repo: Path) -> str:
    head = (repo / ".git" / "HEAD").read_text(encoding="utf-8").strip()
    if not head.startswith("ref:"):
        return head
    rel = head.split(" ", 1)[1].strip()
    pointer = repo / ".git" / rel
    if pointer.is_file():
        return pointer.read_text(encoding="utf-8").strip()
    for line in (repo / ".git" / "packed-refs").read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        sha, ref = line.split(" ", 1)
        if ref.strip() == rel:
            return sha
    raise SystemExit(f"missing ref {rel}")


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def centroid_lps(img_path: Path, mask_path: Path):
    image = nib.load(str(img_path))
    mask = np.asanyarray(nib.load(str(mask_path)).dataobj)
    if image.shape[:3] != mask.shape[:3]:
        raise RuntimeError(f"grid_mismatch img {image.shape} mask {mask.shape}")
    values = set(np.unique(mask).tolist())
    if not values <= {0, 1}:
        raise RuntimeError(f"mask_values {sorted(values)[:8]}")
    binary = mask > 0
    pts = np.argwhere(binary)
    if pts.size == 0:
        raise RuntimeError("empty_mask")
    center = pts.mean(axis=0)
    ras = image.affine @ np.array([center[0], center[1], center[2], 1.0])
    return float(-ras[0]), float(-ras[1]), float(ras[2])


def load_model():
    from monai.networks.nets import resnet50

    from fmcib.models import LoadModel

    trunk = resnet50(
        pretrained=False,
        n_input_channels=1,
        widen_factor=2,
        conv1_t_stride=2,
        feed_forward=False,
        bias_downsample=True,
    )
    model = LoadModel(trunk=trunk, weights_path=str(WEIGHT), heads=[]).to("cuda")
    model.eval()
    return model


def extract_vector(model, csv_path: Path):
    from fmcib.preprocessing import get_dataloader

    loader = get_dataloader(
        str(csv_path),
        batch_size=1,
        num_workers=0,
        spatial_size=(50, 50, 50),
        precropped=False,
    )
    parts = []
    with torch.no_grad():
        for batch in loader:
            parts.append(model(batch.to("cuda")).detach().cpu().numpy())
    if not parts:
        raise RuntimeError("empty_batch")
    return np.concatenate(parts, axis=0).reshape(-1)


def main():
    if torch.__version__ != "2.6.0+cu124":
        raise SystemExit(f"torch {torch.__version__}")
    if not torch.cuda.is_available():
        raise SystemExit("cuda unavailable")
    config = (REPO / ".git" / "config").read_text(encoding="utf-8")
    if "sachinpai/FMCIB" in config or "RenalVision" in config:
        raise SystemExit("unexpected origin")
    commit = read_commit(REPO)
    if commit != EXPECT_COMMIT:
        raise SystemExit(f"commit {commit}")
    if WEIGHT.stat().st_size != EXPECT_SIZE or file_md5(WEIGHT) != EXPECT_MD5:
        raise SystemExit("weight checksum mismatch")
    print(f"commit {commit}", flush=True)
    print(f"weight_md5 {EXPECT_MD5}", flush=True)
    sys.path.insert(0, str(REPO))
    model = load_model()
    ucsf, external = read_cases()
    cases = ucsf + external
    cases.sort(key=lambda case: 0 if case["pid"] == PID else 1)
    out_dir = ROOT / "outputs" / FM
    out_dir.mkdir(parents=True, exist_ok=True)
    coord_dir = out_dir / "coords"
    coord_dir.mkdir(parents=True, exist_ok=True)
    errors = ErrorLog(out_dir / "feature_errors_fmcib.csv")
    store = None
    env = {
        "fm": FM,
        "commit": commit,
        "origin": "https://github.com/AIM-Harvard/foundation-cancer-image-biomarker.git",
        "weight": str(WEIGHT),
        "weight_md5": EXPECT_MD5,
        "weight_bytes": EXPECT_SIZE,
        "torch": torch.__version__,
        "gpu": "CUDA_VISIBLE_DEVICES=0",
        "input": "official 1 mm LPS crop 50x50x50 around mask centroid, 4096-d",
    }
    (out_dir / "env_fmcib.json").write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")

    def one(case, gate: bool) -> bool:
        nonlocal store
        img = Path(case["img"])
        mask = Path(case["mask"])
        try:
            refuse_disallowed([img, mask])
            if not img.is_file() or not mask.is_file():
                raise RuntimeError("missing_file")
            if case["cohort"] == "ucsf" and case.get("qc_label") and case["qc_label"] != case["label"]:
                errors.add(case["pid"], case["cohort"], "label_mismatch", f"qc {case['qc_label']} split {case['label']}")
            x, y, z = centroid_lps(img, mask)
            csv_path = coord_dir / f"{case['cohort']}_{case['pid']}.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["image_path", "coordX", "coordY", "coordZ"])
                writer.writeheader()
                writer.writerow({
                    "image_path": str(img),
                    "coordX": f"{x:.6f}",
                    "coordY": f"{y:.6f}",
                    "coordZ": f"{z:.6f}",
                })
            vector = extract_vector(model, csv_path)
            if vector.shape != (4096,) or not np.isfinite(vector).all():
                raise RuntimeError(f"bad_vector {vector.shape}")
        except Exception as exc:
            reason = str(exc).split(" ", 1)[0]
            if reason not in {"missing_file", "empty_mask", "grid_mismatch", "mask_values", "bad_vector", "protocol_path"}:
                reason = "extract_failed"
            errors.add(case["pid"], case["cohort"], reason, str(exc)[:500])
            if gate:
                print("FULL_GATE_FAIL", case["pid"], exc, flush=True)
            return False
        if store is None:
            store = FeatureStore(ROOT / "features", FM, 4096)
        if not store.has(case["cohort"], case["pid"]):
            store.write(case, vector)
        return True

    if not one(cases[0], True):
        raise SystemExit(1)
    for index, case in enumerate(cases[1:], start=2):
        if store is not None and store.has(case["cohort"], case["pid"]):
            continue
        one(case, False)
        if index % 20 == 0:
            print(f"progress {index}/{len(cases)}", flush=True)
    missing = [case["pid"] for case in cases if store is None or not store.has(case["cohort"], case["pid"])]
    if missing:
        print(f"INCOMPLETE {len(missing)}", flush=True)
        raise SystemExit(2)
    run_probe(ROOT / "features", out_dir, FM)
    print("FMCIB_DONE", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "code"))
    main()
