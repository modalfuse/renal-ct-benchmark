#!/usr/bin/env python3
"""5-class PV-ROI subtype classification with a fixed stratified fold table.

Reads an existing split CSV (pid,label_5class,fold). Does NOT call
StratifiedKFold or otherwise re-draw folds. Seed only affects augmentation
and weight init.

Never reads KiTS23 paths (external test only; see eval_ext.py).

Usage:
  python train_cls.py \\
    --seg-out "$SEG_OUT" \\
    --split "$L4_SPLIT" \\
    --labels "$LABELS" \\
    --out "$OUT" \\
    --seed 20261024 --model resnet18 --size 96 --epochs 100 --bs 8
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

CLASSES = ["ccRCC", "pRCC", "chRCC", "oncocytoma", "AML"]
N_CLASS = len(CLASSES)
FORBIDDEN_PATH_SUBSTR = ("kits23", "kits_23", "kits2023")


def parse_args():
    ap = argparse.ArgumentParser(description="Fixed-fold 5-class CV (no re-split)")
    ap.add_argument("--seg-out", required=True, help="ucsf_pv_seg root with roi_manifest.csv")
    ap.add_argument("--split", required=True, help="ucsf_pv_stratified_split_5fold.csv")
    ap.add_argument("--labels", required=True, help="pid,label_5class CSV")
    ap.add_argument("--out", required=True, help="run directory (must not exist)")
    ap.add_argument("--model", default="resnet18", choices=["resnet18", "resnet34", "densenet121"])
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=20261024)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--run-folds",
        type=int,
        default=0,
        help="0=all folds in split; N=only folds 0..N-1 (smoke)",
    )
    ap.add_argument(
        "--folds-select",
        default="",
        help="comma list of fold ids to run (overrides --run-folds if set)",
    )
    return ap.parse_args()


def refuse_kits_paths(*paths):
    for p in paths:
        s = str(p).lower().replace("\\", "/")
        for bad in FORBIDDEN_PATH_SUBSTR:
            if bad in s:
                raise SystemExit(
                    f"refusing path containing {bad!r} (training must not touch KiTS23): {p}"
                )


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_labels(path: Path) -> dict:
    with path.open(encoding="utf-8-sig") as f:
        return {r["pid"]: r["label_5class"] for r in csv.DictReader(f)}


def load_split(path: Path) -> dict:
    """pid -> {label_5class, fold:int}."""
    out = {}
    with path.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            out[r["pid"]] = {
                "label_5class": r["label_5class"],
                "fold": int(r["fold"]),
            }
    return out


def load_manifest_ok(seg_out: Path) -> list:
    man = seg_out / "roi_manifest.csv"
    with man.open(encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r.get("status") == "ok"]
    return rows


def build_items(seg_out: Path, split: dict, labels: dict) -> list:
    """Return list of dicts: pid, img, mask, y, fold."""
    rows = load_manifest_ok(seg_out)
    items = []
    skipped = {"no_split": 0, "bad_label": 0, "label_mismatch": 0}
    for r in rows:
        pid = r["pid"]
        if pid not in split:
            skipped["no_split"] += 1
            continue
        lab_split = split[pid]["label_5class"]
        lab_csv = labels.get(pid)
        if lab_csv is not None and lab_csv != lab_split:
            skipped["label_mismatch"] += 1
            print(f"WARN label mismatch {pid}: split={lab_split} labels={lab_csv}", flush=True)
        lab = lab_split
        if lab not in CLASSES:
            skipped["bad_label"] += 1
            continue
        img = seg_out / r["img"]
        msk = seg_out / r["mask"]
        if not img.is_file() or not msk.is_file():
            raise SystemExit(f"missing ROI files for {pid}: {img} / {msk}")
        items.append(
            {
                "pid": pid,
                "img": str(img),
                "mask": str(msk),
                "y": CLASSES.index(lab),
                "fold": int(split[pid]["fold"]),
                "label": lab,
            }
        )
    print(
        f"built {len(items)} cases (manifest_ok={len(rows)}; "
        f"skip no_split={skipped['no_split']} bad_label={skipped['bad_label']} "
        f"label_mismatch={skipped['label_mismatch']})",
        flush=True,
    )
    counts = {c: sum(i["y"] == k for i in items) for k, c in enumerate(CLASSES)}
    print("class counts:", counts, flush=True)
    folds = sorted({i["fold"] for i in items})
    print("folds present:", folds, {f: sum(i["fold"] == f for i in items) for f in folds}, flush=True)
    return items


def load_roi(img_path: str, mask_path: str, size: int) -> torch.Tensor:
    x = nib.load(img_path).get_fdata(dtype=np.float32)
    m = (nib.load(mask_path).get_fdata(dtype=np.float32) > 0).astype(np.float32)
    x = (np.clip(x, -200, 300) + 200) / 500.0
    t = torch.from_numpy(np.stack([x, m], axis=0))[None]
    return F.interpolate(t, size=(size,) * 3, mode="trilinear", align_corners=False)[0]


class RoiDataset(torch.utils.data.Dataset):
    def __init__(self, items: list, size: int, aug: bool):
        self.items = items
        self.size = size
        self.aug = aug
        self.cache: dict = {}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        pid = it["pid"]
        if pid not in self.cache:
            self.cache[pid] = load_roi(it["img"], it["mask"], self.size)
        x = self.cache[pid].clone()
        if self.aug:
            for d in (1, 2, 3):
                if random.random() < 0.5:
                    x = x.flip(d)
            x[0] = x[0] * (1 + 0.1 * (random.random() - 0.5)) + 0.05 * (
                random.random() - 0.5
            )
        return x, it["y"], pid


def make_model(name: str) -> nn.Module:
    from monai.networks import nets

    if name == "densenet121":
        return nets.DenseNet121(spatial_dims=3, in_channels=2, out_channels=N_CLASS)
    fn = getattr(nets, name)
    return fn(spatial_dims=3, n_input_channels=2, num_classes=N_CLASS)


def metrics_bundle(y_true: np.ndarray, prob: np.ndarray) -> dict:
    y_pred = prob.argmax(1)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(N_CLASS)), zero_division=0
    )
    try:
        auc = float(
            roc_auc_score(y_true, prob, multi_class="ovr", labels=list(range(N_CLASS)))
        )
    except Exception:
        auc = float("nan")
    per_class = {}
    for k, name in enumerate(CLASSES):
        per_class[name] = {
            "precision": float(prec[k]),
            "recall": float(rec[k]),
            "f1": float(f1[k]),
            "support": int(support[k]),
        }
    return {
        "n": int(len(y_true)),
        "acc": float(accuracy_score(y_true, y_pred)),
        "bacc": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "auc_ovr": auc,
        "per_class": per_class,
        "cm": confusion_matrix(y_true, y_pred, labels=list(range(N_CLASS))).tolist(),
    }


def train_one_fold(
    fold: int,
    items: list,
    args,
    out: Path,
    device: torch.device,
) -> dict:
    tr = [i for i in items if i["fold"] != fold]
    va = [i for i in items if i["fold"] == fold]
    if not tr or not va:
        raise SystemExit(f"fold {fold}: empty train ({len(tr)}) or val ({len(va)})")
    print(f"=== fold {fold}: train={len(tr)} val={len(va)} ===", flush=True)

    y_tr = np.array([i["y"] for i in tr])
    cnt = np.bincount(y_tr, minlength=N_CLASS).astype(float)
    w = np.where(cnt > 0, cnt.sum() / (N_CLASS * np.maximum(cnt, 1.0)), 0.0)
    weight = torch.tensor(w, dtype=torch.float32, device=device)

    dl_tr = torch.utils.data.DataLoader(
        RoiDataset(tr, args.size, True),
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.workers,
        drop_last=len(tr) > args.bs,
    )
    dl_va = torch.utils.data.DataLoader(
        RoiDataset(va, args.size, False),
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.workers,
    )

    model = make_model(args.model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    use_cuda = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
    lossf = nn.CrossEntropyLoss(weight=weight)

    log_path = out / f"fold{fold}_log.csv"
    log = log_path.open("w")
    log.write("epoch,train_loss,val_loss,val_bacc,val_acc,val_macro_f1,sec\n")

    best_bacc = -1.0
    best_prob = None
    best_y = None
    ckpt_path = out / f"fold{fold}_best.pt"

    for ep in range(args.epochs):
        t0 = time.time()
        model.train()
        tl = 0.0
        for x, y, _ in dl_tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_cuda):
                loss = lossf(model(x), y)
            if use_cuda:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            tl += float(loss.item()) * len(y)
        sch.step()

        model.eval()
        P, Y, vl = [], [], 0.0
        with torch.no_grad():
            for x, y, _ in dl_va:
                x = x.to(device)
                y = y.to(device)
                with torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=use_cuda
                ):
                    logits = model(x).float()
                vl += float(F.cross_entropy(logits, y, weight=weight).item()) * len(y)
                P.append(logits.softmax(1).cpu())
                Y.append(y.cpu())
        prob = torch.cat(P).numpy()
        y_np = torch.cat(Y).numpy()
        m = metrics_bundle(y_np, prob)
        sec = time.time() - t0
        log.write(
            f"{ep},{tl / len(tr):.6f},{vl / len(va):.6f},{m['bacc']:.6f},"
            f"{m['acc']:.6f},{m['macro_f1']:.6f},{sec:.1f}\n"
        )
        log.flush()
        mem = (
            f" mem={torch.cuda.max_memory_allocated() / 1e9:.2f}GB"
            if use_cuda
            else ""
        )
        print(
            f"fold{fold} ep{ep} tl={tl / len(tr):.4f} vl={vl / len(va):.4f} "
            f"bacc={m['bacc']:.4f} acc={m['acc']:.4f} mf1={m['macro_f1']:.4f} "
            f"{sec:.1f}s{mem}",
            flush=True,
        )
        if m["bacc"] > best_bacc:
            best_bacc = m["bacc"]
            best_prob = prob
            best_y = y_np
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "fold": fold,
                    "epoch": ep,
                    "bacc": best_bacc,
                    "classes": CLASSES,
                    "model": args.model,
                    "size": args.size,
                    "seed": args.seed,
                },
                ckpt_path,
            )

    log.close()
    assert best_prob is not None and best_y is not None
    final = metrics_bundle(best_y, best_prob)
    final.update(
        {
            "fold": fold,
            "n_train": len(tr),
            "n_val": len(va),
            "best_bacc": float(best_bacc),
            "checkpoint": str(ckpt_path.resolve()),
        }
    )
    # alias keys required by acceptance
    final["balanced_acc"] = final["bacc"]
    print(f"fold{fold} BEST bacc={best_bacc:.4f} macro_f1={final['macro_f1']:.4f}", flush=True)
    return final


def mean_std(vals):
    a = np.array(vals, dtype=float)
    return {"mean": float(np.nanmean(a)), "std": float(np.nanstd(a, ddof=0))}


def main():
    args = parse_args()
    refuse_kits_paths(args.seg_out, args.split, args.labels, args.out)
    seg_out = Path(args.seg_out)
    split_path = Path(args.split)
    labels_path = Path(args.labels)
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"--out already exists (refuse overwrite): {out}")
    out.mkdir(parents=True, exist_ok=False)

    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")

    split = load_split(split_path)
    labels = load_labels(labels_path)
    items = build_items(seg_out, split, labels)
    if len(items) == 0:
        raise SystemExit("no training items after join")

    fold_ids = sorted({i["fold"] for i in items})
    if args.folds_select.strip():
        want = [int(x) for x in args.folds_select.split(",") if x.strip() != ""]
        run_folds = [f for f in want if f in fold_ids]
    elif args.run_folds and args.run_folds > 0:
        run_folds = [f for f in fold_ids if f < args.run_folds]
    else:
        run_folds = fold_ids
    if not run_folds:
        raise SystemExit(f"no folds to run; present={fold_ids}")

    config = {
        "seg_out": str(seg_out.resolve()),
        "split": str(split_path.resolve()),
        "labels": str(labels_path.resolve()),
        "out": str(out.resolve()),
        "model": args.model,
        "size": args.size,
        "epochs": args.epochs,
        "bs": args.bs,
        "lr": args.lr,
        "seed": args.seed,
        "device": args.device,
        "workers": args.workers,
        "n_cases": len(items),
        "folds_present": fold_ids,
        "folds_run": run_folds,
        "classes": CLASSES,
        "hu_clip": [-200, 300],
        "channels": ["hu_scaled", "tumor_mask"],
        "loss": "cross_entropy_inverse_freq",
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "amp": True,
        "split_policy": "fixed_csv_fold_column_no_StratifiedKFold",
        "note": "seed affects aug/init only; folds come from --split CSV",
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print("wrote", out / "config.json", flush=True)
    print("NO StratifiedKFold; folds from CSV:", run_folds, flush=True)

    fold_summaries = []
    for fold in run_folds:
        fold_summaries.append(train_one_fold(fold, items, args, out, device))

    summary = {
        "folds": fold_summaries,
        "aggregate": {
            "bacc": mean_std([f["bacc"] for f in fold_summaries]),
            "balanced_acc": mean_std([f["bacc"] for f in fold_summaries]),
            "acc": mean_std([f["acc"] for f in fold_summaries]),
            "macro_f1": mean_std([f["macro_f1"] for f in fold_summaries]),
            "auc_ovr": mean_std([f["auc_ovr"] for f in fold_summaries]),
        },
        "config_path": str((out / "config.json").resolve()),
        "n_folds_run": len(fold_summaries),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("done", out / "summary.json", flush=True)
    print(
        "aggregate bacc",
        summary["aggregate"]["bacc"],
        "macro_f1",
        summary["aggregate"]["macro_f1"],
        flush=True,
    )


if __name__ == "__main__":
    main()
