#!/usr/bin/env python3
"""External evaluation on RCC-AID / KiTS23 (frozen CV weights).

Loads fold*_best.pt, averages softmax across folds (ensemble), writes metrics.
Inference only — does not train. Training paths must never read KiTS23.

RCC-AID: 3-class RCC only; argmax among three RCC logits; phase strata.
KiTS23 / fiveclass: full 5-class (ccRCC/pRCC/chRCC/oncocytoma/AML).

Usage:
  python eval_ext.py --ckpt-dir ... --dataset rccaid --roi-root ... --out-csv ...
  python eval_ext.py --ckpt-dir ... --dataset kits23 --roi-root ... --out-csv ...
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
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

CLASSES5 = ["ccRCC", "pRCC", "chRCC", "oncocytoma", "AML"]
CLASSES3 = ["ccRCC", "pRCC", "chRCC"]


def parse_args():
    ap = argparse.ArgumentParser(description="External eval (frozen CV weights)")
    ap.add_argument("--ckpt-dir", required=True, help="CV out dir with fold*_best.pt")
    ap.add_argument(
        "--dataset",
        required=True,
        choices=["rccaid", "kits23", "fiveclass"],
        help="fiveclass: same 5-class path as kits23 (no 3-class truncate)",
    )
    ap.add_argument("--roi-root", required=True, help="root with roi_manifest.csv + roi/")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-json", default="", help="optional detailed json")
    ap.add_argument("--cm-png", default="", help="optional confusion-matrix PNG path")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument(
        "--folds",
        default="",
        help="comma fold ids; default=all fold*_best.pt found",
    )
    return ap.parse_args()


def load_roi(img_path: str, mask_path: str, size: int) -> torch.Tensor:
    x = nib.load(img_path).get_fdata(dtype=np.float32)
    m = (nib.load(mask_path).get_fdata(dtype=np.float32) > 0).astype(np.float32)
    x = (np.clip(x, -200, 300) + 200) / 500.0
    t = torch.from_numpy(np.stack([x, m], axis=0))[None]
    return F.interpolate(t, size=(size,) * 3, mode="trilinear", align_corners=False)[0]


def make_model(name: str, n_out: int = 5):
    from monai.networks import nets

    if name == "densenet121":
        return nets.DenseNet121(spatial_dims=3, in_channels=2, out_channels=n_out)
    return getattr(nets, name)(spatial_dims=3, n_input_channels=2, num_classes=n_out)


def discover_ckpts(ckpt_dir: Path, folds_arg: str) -> list:
    if folds_arg.strip():
        folds = [int(x) for x in folds_arg.split(",") if x.strip() != ""]
        paths = []
        for f in folds:
            p = ckpt_dir / f"fold{f}_best.pt"
            if not p.is_file():
                raise SystemExit(f"missing {p}")
            paths.append((f, p))
        return paths
    found = sorted(ckpt_dir.glob("fold*_best.pt"))
    if not found:
        raise SystemExit(f"no fold*_best.pt under {ckpt_dir}")
    out = []
    for p in found:
        # fold0_best.pt
        stem = p.name.replace("_best.pt", "")
        fold = int(stem.replace("fold", ""))
        out.append((fold, p))
    return out


def read_manifest(roi_root: Path, dataset: str) -> list:
    man = roi_root / "roi_manifest.csv"
    with man.open(encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r.get("status", "ok") == "ok"]
    items = []
    for r in rows:
        lab = r["label_5class"]
        if dataset == "rccaid":
            if lab not in CLASSES3:
                continue
            y = CLASSES3.index(lab)
            phase = r.get("phase", "")
        else:
            # kits23 / fiveclass
            if lab not in CLASSES5:
                continue
            y = CLASSES5.index(lab)
            phase = r.get("phase_est", r.get("phase", ""))
        img = roi_root / r["img"]
        msk = roi_root / r["mask"]
        if not img.is_file() or not msk.is_file():
            raise SystemExit(f"missing ROI: {img}")
        items.append(
            {
                "pid": r["pid"],
                "img": str(img),
                "mask": str(msk),
                "y": y,
                "label": lab,
                "phase": phase,
            }
        )
    return items


class ExtDS(torch.utils.data.Dataset):
    def __init__(self, items, size):
        self.items = items
        self.size = size
        self.cache = {}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        if it["pid"] not in self.cache:
            self.cache[it["pid"]] = load_roi(it["img"], it["mask"], self.size)
        return self.cache[it["pid"]], it["y"], it["pid"], it["phase"], it["label"]


@torch.no_grad()
def predict_ensemble(items, ckpts, size, model_name, device, bs, workers):
    models = []
    for fold, path in ckpts:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        sd = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
        mname = (
            blob.get("model", model_name)
            if isinstance(blob, dict)
            else model_name
        )
        size = int(blob.get("size", size)) if isinstance(blob, dict) else size
        model = make_model(mname).to(device)
        model.load_state_dict(sd)
        model.eval()
        models.append(model)
        print(f"loaded fold{fold} {path}", flush=True)

    dl = torch.utils.data.DataLoader(
        ExtDS(items, size), batch_size=bs, shuffle=False, num_workers=workers
    )
    all_prob, all_y, all_pid, all_phase, all_lab = [], [], [], [], []
    use_cuda = device.type == "cuda"
    for x, y, pid, phase, lab in dl:
        x = x.to(device)
        probs = []
        for model in models:
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_cuda):
                logits = model(x).float()
            probs.append(logits.softmax(1))
        prob = torch.stack(probs, 0).mean(0).cpu().numpy()
        all_prob.append(prob)
        all_y.append(y.numpy())
        all_pid.extend(list(pid))
        all_phase.extend(list(phase))
        all_lab.extend(list(lab))
    return (
        np.concatenate(all_prob, 0),
        np.concatenate(all_y, 0),
        all_pid,
        all_phase,
        all_lab,
        size,
    )


def metrics_np(y_true, prob, class_names):
    n = len(class_names)
    y_pred = prob.argmax(1)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n)), zero_division=0
    )
    try:
        auc = float(roc_auc_score(y_true, prob, multi_class="ovr", labels=list(range(n))))
    except Exception:
        auc = float("nan")
    return {
        "n": int(len(y_true)),
        "acc": float(accuracy_score(y_true, y_pred)),
        "bacc": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "auc_ovr": auc,
        "cm": confusion_matrix(y_true, y_pred, labels=list(range(n))).tolist(),
        "per_class": {
            class_names[k]: {
                "precision": float(prec[k]),
                "recall": float(rec[k]),
                "f1": float(f1[k]),
                "support": int(support[k]),
            }
            for k in range(n)
        },
    }


def maybe_cm_png(cm, class_names, path: Path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"skip cm png ({e})", flush=True)
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("pred")
    ax.set_ylabel("true")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print("wrote", path, flush=True)


def write_metrics_csv(path: Path, rows: list):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "subset",
        "n",
        "acc",
        "bacc",
        "macro_f1",
        "auc_ovr",
        "cm_json",
    ]
    # flatten per-class recall into columns
    extra = []
    for name in CLASSES5:
        for m in ("recall", "precision", "f1", "support"):
            extra.append(f"{name}_{m}")
    fields.extend(extra)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            row = {
                "subset": r["subset"],
                "n": r["n"],
                "acc": f"{r['acc']:.6f}",
                "bacc": f"{r['bacc']:.6f}",
                "macro_f1": f"{r['macro_f1']:.6f}",
                "auc_ovr": "" if r["auc_ovr"] != r["auc_ovr"] else f"{r['auc_ovr']:.6f}",
                "cm_json": json.dumps(r["cm"]),
            }
            for name, d in r.get("per_class", {}).items():
                row[f"{name}_recall"] = f"{d['recall']:.6f}"
                row[f"{name}_precision"] = f"{d['precision']:.6f}"
                row[f"{name}_f1"] = f"{d['f1']:.6f}"
                row[f"{name}_support"] = d["support"]
            w.writerow(row)
    print("wrote", path, flush=True)


def main():
    args = parse_args()
    ckpt_dir = Path(args.ckpt_dir)
    roi_root = Path(args.roi_root)
    device = torch.device(args.device)

    cfg_path = ckpt_dir / "config.json"
    model_name = "resnet18"
    size = 96
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text())
        model_name = cfg.get("model", model_name)
        size = int(cfg.get("size", size))

    ckpts = discover_ckpts(ckpt_dir, args.folds)
    items = read_manifest(roi_root, args.dataset)
    print(f"{args.dataset}: {len(items)} cases; {len(ckpts)} checkpoints", flush=True)
    if not items:
        raise SystemExit("no cases")

    prob5, y, pids, phases, labs, size = predict_ensemble(
        items, ckpts, size, model_name, device, args.bs, args.workers
    )

    rows = []
    detail = {
        "dataset": args.dataset,
        "ckpt_dir": str(ckpt_dir),
        "n_models": len(ckpts),
        "n_cases": len(items),
    }
    if args.dataset == "rccaid":
        # Restrict to 3 RCC logits, renormalize
        prob3 = prob5[:, :3]
        prob3 = prob3 / np.clip(prob3.sum(1, keepdims=True), 1e-8, None)
        overall = metrics_np(y, prob3, CLASSES3)
        overall["subset"] = "all"
        rows.append(overall)
        detail["overall"] = overall
        for ph in sorted({p for p in phases if p}):
            idx = [i for i, p in enumerate(phases) if p == ph]
            if not idx:
                continue
            m = metrics_np(y[idx], prob3[idx], CLASSES3)
            m["subset"] = f"phase:{ph}"
            rows.append(m)
            detail.setdefault("by_phase", {})[ph] = m
        class_names = CLASSES3
        cm = overall["cm"]
        phase_col = "phase"
        pred_labels = CLASSES3
        pred_prob = prob3
    else:
        overall = metrics_np(y, prob5, CLASSES5)
        overall["subset"] = "all"
        rows.append(overall)
        detail["overall"] = overall
        for ph in sorted({p for p in phases if p}):
            idx = [i for i, p in enumerate(phases) if p == ph]
            if len(idx) < 2:
                continue
            m = metrics_np(y[idx], prob5[idx], CLASSES5)
            tag = "phase" if args.dataset == "fiveclass" else "phase_est"
            m["subset"] = f"{tag}:{ph}"
            rows.append(m)
        class_names = CLASSES5
        cm = overall["cm"]
        phase_col = "phase" if args.dataset == "fiveclass" else "phase_est"
        pred_labels = CLASSES5
        pred_prob = prob5

    # per-case predictions (sidecar)
    pred_path = Path(args.out_csv).with_name(
        Path(args.out_csv).stem + "_predictions.csv"
    )
    with pred_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["pid", phase_col, "true", "pred"] + [f"p_{c}" for c in pred_labels]
        )
        pred = pred_prob.argmax(1)
        for i in range(len(pids)):
            w.writerow(
                [pids[i], phases[i], labs[i], pred_labels[pred[i]]]
                + [f"{pred_prob[i, k]:.6f}" for k in range(len(pred_labels))]
            )
    print("wrote", pred_path, flush=True)

    write_metrics_csv(Path(args.out_csv), rows)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(detail, indent=2) + "\n")
    cm_png = args.cm_png or str(Path(args.out_csv).with_suffix(".cm.png"))
    maybe_cm_png(cm, class_names, Path(cm_png))


if __name__ == "__main__":
    main()
