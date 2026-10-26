#!/usr/bin/env python3
"""MMM: resize the in-plane mask box to 64x64, keep the slice count, and read Swin encoder features.

Set L4_ROOT. A passed single-case check is followed by the full extraction and linear probe.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from l4_probe import run_probe
from l4_store import ErrorLog, FeatureStore
from l4_tables import read_cases

ROOT = Path(os.environ["L4_ROOT"])
REPO = ROOT / "repos" / "MedicalMultitaskModeling"
ZIP = ROOT / "weights" / "mmm" / "umedpt_checkpoint.zip"
EXPECT_COMMIT = "7ba4fe68c63149832c15f61c32917d4e8593a030"
EXPECT_MD5 = "46c7246f8cbe78f5163786894025c98e"
EXPECT_SIZE = 198578358
FM = "mmm"


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


def load_encoder():
    import io
    import zipfile

    from torchvision.models import swin_s, swin_t

    with zipfile.ZipFile(ZIP) as archive:
        blob = archive.read("umedpt_checkpoint/encoder/module.ckpt")
    state = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=False)["model_state"]
    stripped = {}
    for key, value in state.items():
        name = key
        prefix = "model.wrapped_model."
        if name.startswith(prefix):
            name = name[len(prefix):]
        stripped[name] = value
    stages = {}
    for key in stripped:
        match = re.match(r"features\.(\d+)\.(\d+)\.", key)
        if match:
            stage = int(match.group(1))
            block = int(match.group(2))
            stages[stage] = max(stages.get(stage, -1), block)
    if stages.get(5, -1) >= 17:
        net = swin_s(weights=None)
        arch = "swin_s"
    elif stages.get(5, -1) >= 5:
        net = swin_t(weights=None)
        arch = "swin_t"
    else:
        raise SystemExit(f"unrecognized encoder stages {stages}")
    missing, unexpected = net.load_state_dict(stripped, strict=False)
    # The checkpoint has no LayerNorm parameters. A new module defaults to weight=1 and bias=0; keep those.
    bad = [
        k for k in missing
        if k.startswith("features.") and "norm" not in k and not k.startswith("features.0.2.")
    ]
    extra = list(unexpected)
    print(f"encoder missing_total={len(missing)} missing_other={len(bad)} unexpected={len(extra)}", flush=True)
    if bad or extra:
        raise SystemExit(f"encoder load missing={bad[:8]} unexpected={extra[:8]}")
    net.eval()
    return net, arch, stages


def pool_map(feat: torch.Tensor) -> torch.Tensor:
    if feat.ndim != 4:
        raise RuntimeError(f"feature map {tuple(feat.shape)}")
    if feat.shape[1] >= feat.shape[-1]:
        return feat.mean(dim=(2, 3))
    return feat.mean(dim=(1, 2))


def lesion_cube(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if image.shape != mask.shape:
        raise RuntimeError(f"grid_mismatch img {image.shape} mask {mask.shape}")
    pos = np.argwhere(mask > 0)
    if pos.size == 0:
        raise RuntimeError("empty_mask")
    mins = pos.min(axis=0)
    maxs = pos.max(axis=0) + 1
    slices = tuple(slice(int(a), int(b)) for a, b in zip(mins, maxs))
    cropped = np.asarray(image[slices], dtype=np.float32)
    if min(cropped.shape) < 1:
        raise RuntimeError("empty_crop")
    tensor = torch.from_numpy(np.ascontiguousarray(cropped))[None, None]
    tensor = F.interpolate(tensor, size=(64, 64, cropped.shape[2]), mode="trilinear", align_corners=False)
    cube = tensor[0, 0]
    cube = cube.clamp(-1024, 2048)
    cube = (cube + 1024) / float(2048 + 1024)
    return cube


def embed_cube(net, cube: torch.Tensor) -> np.ndarray:
    planes = cube.permute(2, 0, 1).unsqueeze(1).repeat(1, 3, 1, 1)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=planes.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=planes.dtype).view(1, 3, 1, 1)
    planes = (planes - mean) / std
    parts = []
    with torch.inference_mode():
        for start in range(0, planes.shape[0], 16):
            batch = planes[start:start + 16].to("cuda", non_blocking=True)
            parts.append(pool_map(net.features(batch)).detach().cpu())
    return torch.cat(parts, 0).mean(0).numpy()


def load_pair(case):
    import nibabel as nib

    img_path = Path(case["img"])
    mask_path = Path(case["mask"])
    if not img_path.is_file() or not mask_path.is_file():
        raise RuntimeError("missing_file")
    image = np.asanyarray(nib.load(str(img_path)).dataobj)
    mask = np.asanyarray(nib.load(str(mask_path)).dataobj)
    if image.ndim != 3 or mask.ndim != 3:
        raise RuntimeError(f"ndim img {image.ndim} mask {mask.ndim}")
    values = set(np.unique(mask).tolist())
    if not values <= {0, 1}:
        raise RuntimeError(f"mask_values {sorted(values)[:8]}")
    return image, mask


def main():
    if torch.__version__ != "2.6.0+cu124":
        raise SystemExit(f"torch {torch.__version__}")
    if not torch.cuda.is_available():
        raise SystemExit("cuda unavailable")
    commit = read_commit(REPO)
    if commit != EXPECT_COMMIT:
        raise SystemExit(f"commit {commit}")
    if ZIP.stat().st_size != EXPECT_SIZE or file_md5(ZIP) != EXPECT_MD5:
        raise SystemExit("weight checksum mismatch")
    print(f"commit {commit}", flush=True)
    print(f"weight_md5 {EXPECT_MD5}", flush=True)
    net, arch, stages = load_encoder()
    net = net.to("cuda")
    print(f"encoder {arch} stages {stages}", flush=True)
    ucsf, external = read_cases()
    cases = ucsf + external
    out_dir = ROOT / "outputs" / FM
    out_dir.mkdir(parents=True, exist_ok=True)
    errors = ErrorLog(out_dir / "feature_errors_mmm.csv")
    store = None
    env = {
        "fm": FM,
        "commit": commit,
        "weight": str(ZIP),
        "weight_md5": EXPECT_MD5,
        "weight_bytes": EXPECT_SIZE,
        "torch": torch.__version__,
        "encoder": arch,
        "gpu": "CUDA_VISIBLE_DEVICES",
        "input": "mask bbox, in-plane 64x64, keep Z, portven ROI",
    }
    (out_dir / "env_mmm.json").write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")

    def one(case, gate: bool) -> bool:
        nonlocal store
        try:
            image, mask = load_pair(case)
            if case["cohort"] == "ucsf" and case.get("qc_label") and case["qc_label"] != case["label"]:
                errors.add(case["pid"], case["cohort"], "label_mismatch", f"qc {case['qc_label']} split {case['label']}")
            cube = lesion_cube(image, mask)
            vector = embed_cube(net, cube)
            if not np.isfinite(vector).all():
                raise RuntimeError("non_finite")
        except Exception as exc:
            reason = str(exc).split(" ", 1)[0]
            if reason not in {"missing_file", "empty_mask", "grid_mismatch", "mask_values", "non_finite", "empty_crop"}:
                reason = "extract_failed"
            errors.add(case["pid"], case["cohort"], reason, str(exc)[:500])
            if gate:
                print("SINGLETON_FAIL", case["pid"], exc, flush=True)
            return False
        if store is None:
            store = FeatureStore(ROOT / "features", FM, len(vector))
        if not store.has(case["cohort"], case["pid"]):
            store.write(case, vector)
        if gate:
            payload = {"pid": case["pid"], "dim": int(len(vector)), "finite": True, "cohort": case["cohort"]}
            (out_dir / "singleton.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            print("SINGLETON_OK", json.dumps(payload), flush=True)
        return True

    pending = [case for case in cases if store is None or not store.has(case["cohort"], case["pid"])]
    # The feature width is known only after the first case. FeatureStore is created then.
    if store is None:
        pending = cases
    if not pending:
        run_probe(ROOT / "features", out_dir, FM)
        return
    if not one(pending[0], True):
        raise SystemExit(1)
    for index, case in enumerate(pending[1:], start=2):
        if store is not None and store.has(case["cohort"], case["pid"]):
            continue
        one(case, False)
        if index % 20 == 0:
            print(f"progress {index}/{len(pending)}", flush=True)
    missing = [case["pid"] for case in cases if store is None or not store.has(case["cohort"], case["pid"])]
    if missing:
        print(f"INCOMPLETE {len(missing)}", flush=True)
        raise SystemExit(2)
    run_probe(ROOT / "features", out_dir, FM)
    print("MMM_DONE", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "code"))
    main()
