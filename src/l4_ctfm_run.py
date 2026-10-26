#!/usr/bin/env python3
"""CT-FM official sliding-window encoder, then linear probing.

Set L4_ROOT. The repository under repos/CT-FM must be commit f58c89a and the weight file must be complete.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from l4_probe import run_probe
from l4_store import ErrorLog, FeatureStore
from l4_tables import read_cases

ROOT = Path(os.environ["L4_ROOT"])
REPO = ROOT / "repos" / "CT-FM"
WEIGHT = ROOT / "weights" / "ct_fm_feature_extractor" / "model.safetensors"
CONFIG = ROOT / "weights" / "ct_fm_feature_extractor" / "config.json"
EXPECT_COMMIT = "f58c89a75ff0270ba6e339027e40bf04a221735d"
EXPECT_BYTES = 311116176
FM = "ctfm"


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


def load_model():
    from lighter_zoo import SegResEncoder
    from safetensors.torch import load_file

    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    model = SegResEncoder(
        spatial_dims=int(cfg.get("spatial_dims", 3)),
        in_channels=int(cfg.get("in_channels", 1)),
        init_filters=int(cfg.get("init_filters", 32)),
        blocks_down=tuple(cfg.get("blocks_down", (1, 2, 2, 4, 4))),
        head_module=lambda tensors: torch.nn.functional.adaptive_avg_pool3d(tensors[-1], 1).flatten(1),
    )
    state = load_file(str(WEIGHT))
    candidates = [state]
    for prefix in ("encoder.", "model.", "module.", "state_dict."):
        if any(key.startswith(prefix) for key in state):
            candidates.append({key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)})
    chosen = None
    report = None
    own = model.state_dict()
    for cand in candidates:
        matched = sum(1 for key, value in cand.items() if key in own and tuple(own[key].shape) == tuple(value.shape))
        if chosen is None or matched > chosen[0]:
            chosen = (matched, cand)
    matched, cand = chosen
    if matched < 10:
        sample = list(state)[:12]
        raise SystemExit(f"weight keys do not match encoder, sample={sample}")
    missing, unexpected = model.load_state_dict(cand, strict=False)
    report = {"matched": matched, "missing": len(missing), "unexpected": len(unexpected)}
    print("load", json.dumps(report), flush=True)
    model.eval()
    return model.to("cuda")


def prepare_ucsf(case):
    import h5py
    from monai.data import MetaTensor
    from monai.transforms import Compose, CropForegroundd, Orientationd, ScaleIntensityRanged, Spacingd

    path = Path(case["h5"])
    if not path.is_file():
        raise RuntimeError(f"missing_file {path}")
    spacing = [float(v) for v in str(case["spacing"]).split("x")]
    if len(spacing) != 3 or any(v <= 0 for v in spacing):
        raise RuntimeError(f"bad_spacing {case['spacing']}")
    with h5py.File(path, "r") as handle:
        if "portven" not in handle:
            raise RuntimeError("missing_portven")
        volume = np.asarray(handle["portven"][()], dtype=np.float32)
    affine = torch.eye(4, dtype=torch.float64)
    affine[0, 0] = spacing[0]
    affine[1, 1] = spacing[1]
    affine[2, 2] = spacing[2]
    image = MetaTensor(torch.from_numpy(np.ascontiguousarray(volume))[None], affine=affine)
    pipeline = Compose([
        Orientationd(keys=["image"], axcodes="SPL"),
        Spacingd(keys=["image"], pixdim=[3, 1, 1], mode="bilinear"),
        CropForegroundd(keys=["image"], source_key="image"),
        ScaleIntensityRanged(keys=["image"], a_min=-1024, a_max=2048, b_min=0, b_max=1, clip=True),
    ])
    return pipeline({"image": image})["image"]


def embed(model, image) -> np.ndarray:
    from monai.inferers import SlidingWindowSplitter

    tensor = image.as_tensor() if hasattr(image, "as_tensor") else torch.as_tensor(image)
    if tensor.ndim == 5:
        tensor = tensor[0]
    if tensor.ndim != 4:
        raise RuntimeError(f"image ndim {tensor.ndim}")
    patch = (24, 128, 128)
    pads = [max(0, patch[i] - int(tensor.shape[i + 1])) for i in range(3)]
    padded = False
    if any(pads):
        tensor = torch.nn.functional.pad(tensor, (0, pads[2], 0, pads[1], 0, pads[0]))
        padded = True
    splitter = SlidingWindowSplitter(patch, 0.0)
    feats = []
    with torch.inference_mode():
        for window in splitter(tensor.unsqueeze(0)):
            patch_t = window[0] if isinstance(window, (tuple, list)) else window
            if patch_t.ndim == 4:
                patch_t = patch_t.unsqueeze(0)
            if patch_t.ndim == 6:
                patch_t = patch_t.squeeze(1)
            if patch_t.ndim != 5 or int(patch_t.shape[1]) != 1:
                raise RuntimeError(f"patch {tuple(patch_t.shape)}")
            pred = model(patch_t.to("cuda"))
            if isinstance(pred, (tuple, list)):
                pred = pred[-1]
            feats.append(pred.detach().float().cpu().reshape(-1))
    if not feats:
        raise RuntimeError("no_windows")
    vector = torch.stack(feats, 0).mean(0).numpy()
    return vector, padded, len(feats)


def prepare_nii(path: str):
    from monai.transforms import (
        Compose,
        CropForegroundd,
        EnsureTyped,
        LoadImaged,
        Orientationd,
        ScaleIntensityRanged,
        Spacingd,
    )

    pipeline = Compose([
        LoadImaged(keys=["image"], ensure_channel_first=True),
        EnsureTyped(keys=["image"]),
        Orientationd(keys=["image"], axcodes="SPL"),
        Spacingd(keys=["image"], pixdim=[3, 1, 1], mode="bilinear"),
        CropForegroundd(keys=["image"], source_key="image"),
        ScaleIntensityRanged(keys=["image"], a_min=-1024, a_max=2048, b_min=0, b_max=1, clip=True),
    ])
    return pipeline({"image": path})["image"]


def resolve_commit() -> str:
    try:
        head = read_commit(REPO)
    except Exception as exc:
        raise SystemExit(f"CT-FM HEAD unreadable: {exc}") from exc
    print("commit", head, flush=True)
    if head != EXPECT_COMMIT:
        raise SystemExit(f"CT-FM HEAD not {EXPECT_COMMIT}: {head}")
    return head


def main():
    if torch.__version__ != "2.6.0+cu124":
        raise SystemExit(f"torch {torch.__version__}")
    if not WEIGHT.is_file() or WEIGHT.stat().st_size != EXPECT_BYTES:
        raise SystemExit(f"weight bytes {WEIGHT.stat().st_size if WEIGHT.is_file() else 'missing'}")
    if (WEIGHT.parent / "model.safetensors.part").is_file() and not WEIGHT.is_file():
        raise SystemExit("model.safetensors still incomplete")
    commit = resolve_commit()
    print(f"commit {commit}", flush=True)
    ucsf, external = read_cases()
    cases = ucsf + external
    out_dir = ROOT / "outputs" / FM
    out_dir.mkdir(parents=True, exist_ok=True)
    errors = ErrorLog(out_dir / "feature_errors_ctfm.csv")
    model = load_model()
    store = None

    def one(case, gate: bool) -> bool:
        nonlocal store
        try:
            if case["cohort"] == "ucsf":
                image = prepare_ucsf(case)
            else:
                if not Path(case["img"]).is_file():
                    raise RuntimeError("missing_file")
                image = prepare_nii(case["img"])
            vector, padded, nwin = embed(model, image)
            if padded:
                errors.add(case["pid"], case["cohort"], "padded_to_patch", "spatial size below 24x128x128")
            if not np.isfinite(vector).all():
                raise RuntimeError("non_finite")
        except Exception as exc:
            errors.add(case["pid"], case["cohort"], "extract_failed", str(exc)[:500])
            if gate:
                print("SINGLETON_FAIL", case["pid"], exc, flush=True)
            return False
        if store is None:
            store = FeatureStore(ROOT / "features", FM, len(vector))
        if not store.has(case["cohort"], case["pid"]):
            store.write(case, vector)
        if gate:
            payload = {
                "pid": case["pid"],
                "dim": int(len(vector)),
                "n_windows": int(nwin),
                "padded": bool(padded),
                "finite": True,
                "commit": commit,
                "weight_bytes": EXPECT_BYTES,
            }
            (out_dir / "singleton.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            print("SINGLETON_OK", json.dumps(payload), flush=True)
        return True

    if not one(cases[0], True):
        raise SystemExit(1)
    for index, case in enumerate(cases[1:], start=2):
        if store is not None and store.has(case["cohort"], case["pid"]):
            continue
        one(case, False)
        if index % 10 == 0:
            print(f"progress {index}/{len(cases)}", flush=True)
    missing = [case["pid"] for case in cases if store is None or not store.has(case["cohort"], case["pid"])]
    if missing:
        print(f"INCOMPLETE {len(missing)}", flush=True)
        raise SystemExit(2)
    run_probe(ROOT / "features", out_dir, FM)
    print("CTFM_DONE", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "code"))
    main()
