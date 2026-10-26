#!/usr/bin/env python3
"""L3 backbone / class-rebalance CV (ResNet50 × 3 frozen runs).

New file only — does not modify train_cls.py / eval_ext.py.
Folds come from a fixed CSV (no StratifiedKFold). Seed affects aug/init only.
Training must never read KiTS23 paths.

Runs (--run):
  resnet50         : MONAI ResNet50 + inverse-freq weighted CE (same as ResNet18)
  resnet50_focal   : MONAI ResNet50 + multiclass focal γ=2, α=w
  resnet50_sampler : MONAI ResNet50 + unweighted CE + WeightedRandomSampler(w)

Usage:
  python l3_backbone.py --run resnet50 \\
    --split "$L4_SPLIT" \\
    --labels "$LABELS" \\
    --seg-out "$SEG_OUT" \\
    --out "$OUT" \\
    --seed 20261024
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
RUN_CHOICES = ("resnet50", "resnet50_focal", "resnet50_sampler")


def parse_args():
    ap = argparse.ArgumentParser(description="L3 ResNet50 CV (fixed folds, 3 runs)")
    ap.add_argument("--seg-out", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True, help="run directory")
    ap.add_argument("--run", required=True, choices=RUN_CHOICES)
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=20261024)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--run-folds", type=int, default=0, help="0=all; N=folds 0..N-1")
    ap.add_argument("--folds-select", default="", help="comma fold ids (overrides --run-folds)")
    ap.add_argument(
        "--allow-existing-out",
        action="store_true",
        help="allow writing into an existing --out (for multi-GPU fold workers)",
    )
    ap.add_argument(
        "--aggregate-only",
        action="store_true",
        help="only rebuild cv_fold_metrics.csv / cv_aggregate.json from fold*_metrics.json",
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
    out = {}
    with path.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            out[r["pid"]] = {"label_5class": r["label_5class"], "fold": int(r["fold"])}
    return out


def load_manifest_ok(seg_out: Path) -> list:
    man = seg_out / "roi_manifest.csv"
    with man.open(encoding="utf-8-sig") as f:
        return [r for r in csv.DictReader(f) if r.get("status") == "ok"]


def build_items(seg_out: Path, split: dict, labels: dict) -> list:
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
            x[0] = x[0] * (1 + 0.1 * (random.random() - 0.5)) + 0.05 * (random.random() - 0.5)
        return x, it["y"], pid


def make_model() -> nn.Module:
    from monai.networks.nets import resnet50

    return resnet50(spatial_dims=3, n_input_channels=2, num_classes=N_CLASS)


class FocalLoss(nn.Module):
    """Multiclass focal: -α_y (1-p_t)^γ log p_t, mean over batch."""

    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=1)
        p = log_p.exp()
        log_pt = log_p.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = p.gather(1, target.unsqueeze(1)).squeeze(1)
        at = self.alpha.gather(0, target)
        loss = -at * (1.0 - pt).pow(self.gamma) * log_pt
        return loss.mean()


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


def inverse_freq_weights(y_tr: np.ndarray) -> np.ndarray:
    cnt = np.bincount(y_tr, minlength=N_CLASS).astype(float)
    n = float(cnt.sum())
    w = np.zeros(N_CLASS, dtype=float)
    for c in range(N_CLASS):
        if cnt[c] > 0:
            w[c] = n / (N_CLASS * cnt[c])
    return w, cnt


def train_one_fold(fold: int, items: list, args, out: Path, device: torch.device) -> dict:
    tr = [i for i in items if i["fold"] != fold]
    va = [i for i in items if i["fold"] == fold]
    if not tr or not va:
        raise SystemExit(f"fold {fold}: empty train ({len(tr)}) or val ({len(va)})")
    print(f"=== fold {fold}: train={len(tr)} val={len(va)} run={args.run} ===", flush=True)

    y_tr = np.array([i["y"] for i in tr])
    w, cnt = inverse_freq_weights(y_tr)
    weight = torch.tensor(w, dtype=torch.float32, device=device)
    print(f"fold{fold} n_c={cnt.tolist()} w_c={w.tolist()}", flush=True)

    ds_tr = RoiDataset(tr, args.size, True)
    ds_va = RoiDataset(va, args.size, False)

    if args.run == "resnet50_sampler":
        sample_w = torch.tensor([w[i["y"]] for i in tr], dtype=torch.double)
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_w, num_samples=len(tr), replacement=True
        )
        dl_tr = torch.utils.data.DataLoader(
            ds_tr,
            batch_size=args.bs,
            sampler=sampler,
            shuffle=False,
            num_workers=args.workers,
            drop_last=len(tr) > args.bs,
        )
    else:
        dl_tr = torch.utils.data.DataLoader(
            ds_tr,
            batch_size=args.bs,
            shuffle=True,
            num_workers=args.workers,
            drop_last=len(tr) > args.bs,
        )
    dl_va = torch.utils.data.DataLoader(
        ds_va, batch_size=args.bs, shuffle=False, num_workers=args.workers
    )

    model = make_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    use_cuda = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    if args.run == "resnet50":
        lossf = nn.CrossEntropyLoss(weight=weight)
        loss_name = "cross_entropy_inverse_freq"
    elif args.run == "resnet50_focal":
        lossf = FocalLoss(alpha=weight, gamma=2.0)
        loss_name = "focal_gamma2_alpha_inverse_freq"
    else:
        lossf = nn.CrossEntropyLoss(weight=None)
        loss_name = "cross_entropy_unweighted_plus_WeightedRandomSampler"

    log_path = out / f"fold{fold}_log.csv"
    log = log_path.open("w")
    log.write("epoch,train_loss,val_loss,val_bacc,val_acc,val_macro_f1,sec\n")

    best_bacc = -1.0
    best_prob = None
    best_y = None
    ckpt_path = out / f"fold{fold}_best.pt"
    peak_mem = 0.0
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    t_fold0 = time.time()
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
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_cuda):
                    logits = model(x).float()
                # val loss: CE for reporting (weighted when R1; unweighted for R2/R3)
                if args.run == "resnet50":
                    vl += float(F.cross_entropy(logits, y, weight=weight).item()) * len(y)
                else:
                    vl += float(F.cross_entropy(logits, y).item()) * len(y)
                P.append(logits.softmax(1).cpu())
                Y.append(y.cpu())
        prob = torch.cat(P).numpy()
        y_np = torch.cat(Y).numpy()
        m = metrics_bundle(y_np, prob)
        sec = time.time() - t0
        if use_cuda:
            peak_mem = max(peak_mem, torch.cuda.max_memory_allocated() / 1e9)
        log.write(
            f"{ep},{tl / len(tr):.6f},{vl / len(va):.6f},{m['bacc']:.6f},"
            f"{m['acc']:.6f},{m['macro_f1']:.6f},{sec:.1f}\n"
        )
        log.flush()
        mem = f" mem={peak_mem:.2f}GB" if use_cuda else ""
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
                    "model": "resnet50",
                    "size": args.size,
                    "seed": args.seed,
                    "run": args.run,
                },
                ckpt_path,
            )

    log.close()
    assert best_prob is not None and best_y is not None
    final = metrics_bundle(best_y, best_prob)
    wall = time.time() - t_fold0
    final.update(
        {
            "fold": fold,
            "n_train": len(tr),
            "n_val": len(va),
            "best_bacc": float(best_bacc),
            "checkpoint": str(ckpt_path.resolve()),
            "balanced_acc": final["bacc"],
            "n_c": {CLASSES[k]: int(cnt[k]) for k in range(N_CLASS)},
            "w_c": {CLASSES[k]: float(w[k]) for k in range(N_CLASS)},
            "loss": loss_name,
            "peak_mem_gb": float(peak_mem),
            "wall_sec": float(wall),
        }
    )
    (out / f"fold{fold}_metrics.json").write_text(json.dumps(final, indent=2) + "\n")
    print(f"fold{fold} BEST bacc={best_bacc:.4f} macro_f1={final['macro_f1']:.4f} wall={wall:.0f}s", flush=True)
    return final


def mean_std(vals):
    a = np.array(vals, dtype=float)
    return {"mean": float(np.nanmean(a)), "std": float(np.nanstd(a, ddof=0))}


def write_cv_tables(out: Path, fold_summaries: list, args, config: dict, wall_total: float):
    # cv_fold_metrics.csv
    fields = [
        "fold",
        "n_train",
        "n_val",
        "acc",
        "bacc",
        "macro_f1",
        "auc_ovr",
        "cm_json",
    ]
    for name in CLASSES:
        for m in ("recall", "precision", "f1", "support"):
            fields.append(f"{name}_{m}")
    csv_path = out / "cv_fold_metrics.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for fs in sorted(fold_summaries, key=lambda x: x["fold"]):
            row = {
                "fold": fs["fold"],
                "n_train": fs["n_train"],
                "n_val": fs["n_val"],
                "acc": f"{fs['acc']:.6f}",
                "bacc": f"{fs['bacc']:.6f}",
                "macro_f1": f"{fs['macro_f1']:.6f}",
                "auc_ovr": "" if fs["auc_ovr"] != fs["auc_ovr"] else f"{fs['auc_ovr']:.6f}",
                "cm_json": json.dumps(fs["cm"]),
            }
            for name, d in fs["per_class"].items():
                row[f"{name}_recall"] = f"{d['recall']:.6f}"
                row[f"{name}_precision"] = f"{d['precision']:.6f}"
                row[f"{name}_f1"] = f"{d['f1']:.6f}"
                row[f"{name}_support"] = d["support"]
            w.writerow(row)
    print("wrote", csv_path, flush=True)

    agg = {
        "run": args.run,
        "model": "resnet50",
        "n_folds": len(fold_summaries),
        "aggregate": {
            "bacc": mean_std([f["bacc"] for f in fold_summaries]),
            "acc": mean_std([f["acc"] for f in fold_summaries]),
            "macro_f1": mean_std([f["macro_f1"] for f in fold_summaries]),
            "auc_ovr": mean_std([f["auc_ovr"] for f in fold_summaries]),
        },
        "folds": [
            {
                "fold": f["fold"],
                "n_train": f["n_train"],
                "n_val": f["n_val"],
                "bacc": f["bacc"],
                "acc": f["acc"],
                "macro_f1": f["macro_f1"],
                "auc_ovr": f["auc_ovr"],
                "n_c": f.get("n_c"),
                "w_c": f.get("w_c"),
                "peak_mem_gb": f.get("peak_mem_gb"),
                "wall_sec": f.get("wall_sec"),
                "best_epoch_bacc": f.get("best_bacc"),
                "per_class": f.get("per_class"),
                "cm": f.get("cm"),
            }
            for f in sorted(fold_summaries, key=lambda x: x["fold"])
        ],
        "loss_or_sampler": {
            "resnet50": "CrossEntropyLoss(weight=w), w_c=n/(5 n_c) on fold train only",
            "resnet50_focal": "focal gamma=2, alpha_c=w_c",
            "resnet50_sampler": "CE unweighted + WeightedRandomSampler(w_c), num_samples=n_train, replacement=True",
        }[args.run],
        "peak_mem_gb_max": float(max((f.get("peak_mem_gb") or 0) for f in fold_summaries)),
        "wall_sec_total_sum_folds": float(sum((f.get("wall_sec") or 0) for f in fold_summaries)),
        "wall_sec_aggregate_pass": float(wall_total),
        "config": config,
        "split_policy": "fixed_csv_fold_column_no_StratifiedKFold",
        "note": "seed affects aug/init only; folds from --split CSV; no KiTS23 in training",
    }
    (out / "cv_aggregate.json").write_text(json.dumps(agg, indent=2) + "\n")
    print("wrote", out / "cv_aggregate.json", flush=True)
    print("aggregate bacc", agg["aggregate"]["bacc"], flush=True)


def load_fold_metrics(out: Path) -> list:
    paths = sorted(out.glob("fold*_metrics.json"))
    out_list = []
    for p in paths:
        out_list.append(json.loads(p.read_text()))
    return out_list


def main():
    args = parse_args()
    refuse_kits_paths(args.seg_out, args.split, args.labels, args.out)
    seg_out = Path(args.seg_out)
    split_path = Path(args.split)
    labels_path = Path(args.labels)
    out = Path(args.out)

    if args.aggregate_only:
        if not out.is_dir():
            raise SystemExit(f"--out missing for aggregate-only: {out}")
        cfg = {}
        cfg_path = out / "config.json"
        if cfg_path.is_file():
            cfg = json.loads(cfg_path.read_text())
        folds = load_fold_metrics(out)
        if len(folds) < 1:
            raise SystemExit(f"no fold*_metrics.json under {out}")
        write_cv_tables(out, folds, args, cfg, wall_total=0.0)
        return

    if out.exists() and not args.allow_existing_out:
        raise SystemExit(f"--out already exists (refuse overwrite): {out}")
    out.mkdir(parents=True, exist_ok=True)

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
        "run": args.run,
        "model": "resnet50",
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
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "amp": True,
        "split_policy": "fixed_csv_fold_column_no_StratifiedKFold",
        "note": "seed affects aug/init only; folds come from --split CSV",
    }
    # merge folds_run if parallel workers
    cfg_path = out / "config.json"
    if cfg_path.is_file() and args.allow_existing_out:
        old = json.loads(cfg_path.read_text())
        merged = sorted(set(old.get("folds_run", []) + run_folds))
        config["folds_run"] = merged
        old.update(config)
        config = old
    cfg_path.write_text(json.dumps(config, indent=2) + "\n")
    print("wrote", cfg_path, flush=True)
    print("NO StratifiedKFold; folds from CSV:", run_folds, "run=", args.run, flush=True)

    t0 = time.time()
    fold_summaries = []
    for fold in run_folds:
        fold_summaries.append(train_one_fold(fold, items, args, out, device))

    # If this worker did not run all folds, only write per-fold metrics; leave aggregate to --aggregate-only
    all_metrics = load_fold_metrics(out)
    present = sorted(f["fold"] for f in all_metrics)
    if present == fold_ids:
        write_cv_tables(out, all_metrics, args, config, wall_total=time.time() - t0)
    else:
        print(
            f"partial folds present={present} expected={fold_ids}; skip aggregate (use --aggregate-only later)",
            flush=True,
        )


if __name__ == "__main__":
    main()
