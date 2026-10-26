#!/usr/bin/env python3
"""Size-only multinomial logistic baseline (log volume, log long diameter).

Cross-validation uses a fixed fold CSV. External probabilities are the mean of
five fold-wise models. RCC-AID metrics use the first three classes, renormalized.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

CLASSES5 = ["ccRCC", "pRCC", "chRCC", "oncocytoma", "AML"]
CLASSES3 = ["ccRCC", "pRCC", "chRCC"]
FEATS = ["log_volume", "log_long"]
N_SPLIT = 532
N_RCC = 91
N_KITS = 427


def parse_args():
    p = argparse.ArgumentParser(description="L0 size-only logistic baseline")
    p.add_argument("--split", required=True, type=Path)
    p.add_argument("--qc-seg", required=True, type=Path)
    p.add_argument("--ucsf-long-cm", required=True, type=Path)
    p.add_argument("--ucsf-ref", required=True, type=Path, help="pid,tumor_size_cm reference column only")
    p.add_argument("--rcc-cases", required=True, type=Path)
    p.add_argument("--rcc-roi", required=True, type=Path)
    p.add_argument("--kits-cases", required=True, type=Path)
    p.add_argument("--kits-roi", required=True, type=Path)
    p.add_argument("--resnet-rcc", required=True, type=Path)
    p.add_argument("--resnet-kits", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    return p.parse_args()


def refuse_disallowed(paths):
    token = "batch" + "1"
    for path in paths:
        if token in str(path).lower():
            raise SystemExit("an input path is outside this protocol")


def long_cm_from_mask(path: Path) -> float:
    img = nib.load(str(path))
    mask = np.asarray(img.dataobj) > 0
    if not mask.any():
        return float("nan")
    zooms = np.asarray(img.header.get_zooms()[:3], dtype=np.float64)
    idx = np.nonzero(mask)
    edges = [
        (int(idx[ax].max()) - int(idx[ax].min()) + 1) * float(zooms[ax]) for ax in range(3)
    ]
    return float(max(edges) / 10.0)


def require_ids(name: str, expected: set, got: set, out: Path):
    missing = sorted(expected - got)
    extra = sorted(got - expected)
    if not missing and not extra:
        return
    path = out / f"missing_{name}.txt"
    path.write_text(
        "missing\n" + "\n".join(missing) + "\nextra\n" + "\n".join(extra) + "\n",
        encoding="utf-8",
    )
    print(f"id mismatch {name}: missing={len(missing)} extra={len(extra)} -> {path.name}", flush=True)
    raise SystemExit(1)


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
        "y_pred": y_pred,
    }


def mean_std(vals):
    a = np.array(vals, dtype=float)
    return {"mean": float(np.nanmean(a)), "std": float(np.nanstd(a, ddof=0))}


def fmt_num(v, nd=6):
    if v != v:
        return ""
    return f"{v:.{nd}f}"


def write_metrics_csv(path: Path, rows: list):
    fields = ["subset", "n", "acc", "bacc", "macro_f1", "auc_ovr", "cm_json"]
    for name in CLASSES5:
        for metric in ("recall", "precision", "f1", "support"):
            fields.append(f"{name}_{metric}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        import csv

        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = {
                "subset": row["subset"],
                "n": row["n"],
                "acc": f"{row['acc']:.6f}",
                "bacc": f"{row['bacc']:.6f}",
                "macro_f1": f"{row['macro_f1']:.6f}",
                "auc_ovr": fmt_num(row["auc_ovr"]),
                "cm_json": json.dumps(row["cm"]),
            }
            for name, detail in row.get("per_class", {}).items():
                out[f"{name}_recall"] = f"{detail['recall']:.6f}"
                out[f"{name}_precision"] = f"{detail['precision']:.6f}"
                out[f"{name}_f1"] = f"{detail['f1']:.6f}"
                out[f"{name}_support"] = detail["support"]
            writer.writerow(out)
    print("wrote", path, flush=True)


def write_predictions(path: Path, phase_col: str, pids, phases, labels, pred_names, prob):
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    pred = prob.argmax(1)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pid", phase_col, "true", "pred"] + [f"p_{c}" for c in pred_names])
        for i, pid in enumerate(pids):
            writer.writerow(
                [pid, phases[i], labels[i], pred_names[int(pred[i])]]
                + [f"{prob[i, k]:.6f}" for k in range(len(pred_names))]
            )
    print("wrote", path, flush=True)


def maybe_cm_png(cm, class_names, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(np.asarray(cm), cmap="Blues")
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


def json_safe(obj):
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, float) and obj != obj:
        return None
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if v != v else v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def fit_fold(train_df):
    scaler = StandardScaler()
    x = scaler.fit_transform(train_df[FEATS].to_numpy(dtype=float))
    if x.shape[1] != 2:
        raise SystemExit("feature count is not 2")
    clf = LogisticRegression(
        solver="lbfgs",
        class_weight="balanced",
        max_iter=2000,
        C=1.0,
        random_state=20261024,
    )
    clf.fit(x, train_df["y"].to_numpy(dtype=int))
    got = [int(c) for c in clf.classes_]
    if got != [0, 1, 2, 3, 4]:
        raise SystemExit(f"class order {got}")
    return scaler, clf


def proba_of(scaler, clf, frame):
    x = scaler.transform(frame[FEATS].to_numpy(dtype=float))
    return clf.predict_proba(x)


def attach_long_cm(frame, roi_dir: Path, id_col: str):
    values = []
    missing = []
    for pid in frame[id_col].astype(str):
        path = roi_dir / f"{pid}_mask.nii.gz"
        if not path.is_file():
            missing.append(pid)
            values.append(np.nan)
            continue
        values.append(long_cm_from_mask(path))
    if missing:
        raise SystemExit(f"missing masks: {len(missing)} (first {missing[:5]})")
    frame = frame.copy()
    frame["long_cm"] = values
    frame["long_cm_source"] = "mask_aabb"
    return frame


def load_ucsf(args, out: Path) -> pd.DataFrame:
    split = pd.read_csv(args.split, dtype={"pid": str, "label_5class": str})
    if list(split.columns[:3]) != ["pid", "label_5class", "fold"]:
        raise SystemExit(f"unexpected split columns {list(split.columns)}")
    qc = pd.read_csv(args.qc_seg, dtype={"pid": str})
    qc = qc[qc["status"].astype(str) == "ok"]
    long_cm = pd.read_csv(args.ucsf_long_cm, dtype={"pid": str})
    ref = pd.read_csv(args.ucsf_ref, dtype={"pid": str})
    require_ids("ucsf_qc", set(split["pid"]), set(qc["pid"]), out)
    require_ids("ucsf_long_cm", set(split["pid"]), set(long_cm["pid"]), out)
    merged = split.merge(qc[["pid", "mask_ml", "label_5class"]], on="pid", suffixes=("", "_qc"))
    if (merged["label_5class"] != merged["label_5class_qc"]).any():
        raise SystemExit("split label differs from qc label")
    merged = merged.merge(long_cm, on="pid", how="left")
    ref = ref[["pid", "tumor_size_cm"]].drop_duplicates("pid")
    merged = merged.merge(ref, on="pid", how="left")
    if len(merged) != len(split):
        raise SystemExit(f"ucsf row count {len(merged)} != {len(split)}")
    unknown = sorted(set(merged["label_5class"]) - set(CLASSES5))
    if unknown:
        raise SystemExit(f"unknown labels {unknown}")
    merged["y"] = merged["label_5class"].map({n: i for i, n in enumerate(CLASSES5)}).astype(int)
    merged["volume_ml"] = pd.to_numeric(merged["mask_ml"], errors="coerce")
    merged["long_cm"] = pd.to_numeric(merged["long_cm"], errors="coerce")
    merged["ref_diameter_cm"] = pd.to_numeric(merged["tumor_size_cm"], errors="coerce")
    merged["cohort"] = "ucsf"
    merged["phase"] = "portven"
    merged["long_cm_source"] = "mask_aabb_read_only"
    return merged


def load_rcc(args) -> pd.DataFrame:
    cases = pd.read_csv(args.rcc_cases, dtype=str)
    flag = cases["selected_external"].astype(str).str.lower().isin(["true", "1"])
    sel = cases.loc[flag].copy()
    sel["pid"] = sel["file"].astype(str)
    sel["label_5class"] = sel["subtype"].astype(str)
    unknown = sorted(set(sel["label_5class"]) - set(CLASSES3))
    if unknown:
        raise SystemExit(f"rcc labels outside three classes {unknown}")
    sel["y"] = sel["label_5class"].map({n: i for i, n in enumerate(CLASSES5)}).astype(int)
    sel["volume_ml"] = pd.to_numeric(sel["tumor_ml"], errors="coerce")
    sel["phase"] = sel["phase"].fillna("").astype(str)
    sel["fold"] = np.nan
    sel["cohort"] = "rccaid"
    sel["ref_diameter_cm"] = np.nan
    sel = attach_long_cm(sel, args.rcc_roi, "pid")
    return sel


def load_kits(args) -> pd.DataFrame:
    cases = pd.read_csv(args.kits_cases, dtype=str)
    ok = cases[cases["status"].astype(str) == "ok"].copy()
    ok["pid"] = ok["pid"].astype(str)
    ok["label_5class"] = ok["label_5class"].astype(str)
    unknown = sorted(set(ok["label_5class"]) - set(CLASSES5))
    if unknown:
        raise SystemExit(f"kits labels outside five classes {unknown}")
    ok["y"] = ok["label_5class"].map({n: i for i, n in enumerate(CLASSES5)}).astype(int)
    ok["volume_ml"] = pd.to_numeric(ok["tumor_vol_ml"], errors="coerce")
    ok["phase"] = ok["phase_est"].fillna("").astype(str)
    ok["fold"] = np.nan
    ok["cohort"] = "kits23"
    ok["ref_diameter_cm"] = pd.to_numeric(ok["radiographic_size_cm"], errors="coerce")
    ok = attach_long_cm(ok, args.kits_roi, "pid")
    return ok


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    bad = (out["volume_ml"] <= 0) | (out["long_cm"] <= 0) | out["volume_ml"].isna() | out["long_cm"].isna()
    if bad.any():
        pids = out.loc[bad, "pid"].astype(str).tolist()
        raise SystemExit(f"nonpositive size features: {len(pids)} (first {pids[:8]})")
    out["log_volume"] = np.log(out["volume_ml"].to_numpy(dtype=float))
    out["log_long"] = np.log(out["long_cm"].to_numpy(dtype=float))
    return out


def write_features(path: Path, frames):
    cols = [
        "pid",
        "cohort",
        "label_5class",
        "fold",
        "phase",
        "volume_ml",
        "long_cm",
        "log_volume",
        "log_long",
        "ref_diameter_cm",
        "long_cm_source",
    ]
    all_df = pd.concat([df.copy() for df in frames], ignore_index=True)
    view = all_df[cols].rename(columns={"label_5class": "label"})
    fold = pd.Series("", index=view.index, dtype=object)
    ucsf_row = view["cohort"] == "ucsf"
    fold.loc[ucsf_row] = view.loc[ucsf_row, "fold"].astype(int).astype(str)
    view["fold"] = fold
    view.to_csv(path, index=False)
    print("wrote", path, flush=True)
    return all_df


def support_counts(labels):
    s = pd.Series(labels).value_counts()
    return {name: int(s.get(name, 0)) for name in CLASSES5}


def bacc_labels(y_true, y_pred):
    if len(y_true) == 0:
        return float("nan")
    return float(balanced_accuracy_score(np.asarray(y_true), np.asarray(y_pred)))


def tertile_table(ucsf_oof, rcc_pred, kits_pred, resnet_rcc, resnet_kits, ucsf_max):
    rows = []

    def add_within(cohort, frame, pred_col, resnet_col, note):
        cat, edges = assign_tertiles(frame["long_cm"])
        work = frame.copy()
        work["bin"] = cat.astype(str)
        edge_txt = ",".join(f"{v:.6f}" for v in edges)
        order = [b for b in ["T1", "T2", "T3"] if b in set(work["bin"])]
        extra = [b for b in work["bin"].unique() if b not in order and b != "nan"]
        for i, bin_name in enumerate(order + extra):
            sub = work[work["bin"] == bin_name]
            lo = float(edges[i]) if i < len(edges) else float("nan")
            hi = float(edges[i + 1]) if i + 1 < len(edges) else float("nan")
            rec = {
                "section": "within_tertile",
                "cohort": cohort,
                "bin": bin_name,
                "n": int(len(sub)),
                "long_cm_lo": lo,
                "long_cm_hi": hi,
                "edges_cm": edge_txt,
                "ucsf_long_cm_max": ucsf_max,
                "bacc_l0": bacc_labels(sub["label_5class"], sub[pred_col]),
                "bacc_resnet": (
                    float("nan")
                    if resnet_col is None
                    else bacc_labels(sub["label_5class"], sub[resnet_col])
                ),
                "note": note,
            }
            rec.update({f"{k}_support": v for k, v in support_counts(sub["label_5class"]).items()})
            rows.append(rec)

    add_within(
        "ucsf",
        ucsf_oof,
        "pred",
        None,
        "ResNet out-of-fold predictions were not saved. No GPU inference was run.",
    )
    add_within("rccaid", rcc_pred, "pred", "pred_resnet", "same long_cm edges for L0 and ResNet")
    add_within("kits23", kits_pred, "pred", "pred_resnet", "same long_cm edges for L0 and ResNet")

    for cohort, frame in (("rccaid", rcc_pred), ("kits23", kits_pred)):
        for bin_name, mask in (
            ("le_ucsf_max", frame["long_cm"] <= ucsf_max),
            ("gt_ucsf_max", frame["long_cm"] > ucsf_max),
        ):
            sub = frame.loc[mask]
            rec = {
                "section": "vs_ucsf_max",
                "cohort": cohort,
                "bin": bin_name,
                "n": int(len(sub)),
                "long_cm_lo": float(sub["long_cm"].min()) if len(sub) else float("nan"),
                "long_cm_hi": float(sub["long_cm"].max()) if len(sub) else float("nan"),
                "edges_cm": f"{ucsf_max:.6f}",
                "ucsf_long_cm_max": ucsf_max,
                "bacc_l0": bacc_labels(sub["label_5class"], sub["pred"]),
                "bacc_resnet": bacc_labels(sub["label_5class"], sub["pred_resnet"]),
                "note": "cut is the maximum UCSF mask long_cm",
            }
            rec.update({f"{k}_support": v for k, v in support_counts(sub["label_5class"]).items()})
            rows.append(rec)
    return pd.DataFrame(rows)


def assign_tertiles(series: pd.Series):
    try:
        cat, edges = pd.qcut(
            series, 3, labels=["T1", "T2", "T3"], retbins=True, duplicates="drop"
        )
        return cat, edges
    except ValueError:
        cat, edges = pd.qcut(series, 3, retbins=True, duplicates="drop")
        names = [f"T{i + 1}" for i in range(len(edges) - 1)]
        cat = pd.qcut(series, 3, labels=names, retbins=False, duplicates="drop")
        return cat, edges


def join_resnet(frame, pred_l0, resnet_path: Path, phase_col: str):
    rn = pd.read_csv(resnet_path, dtype=str)
    rn["pid"] = rn["pid"].astype(str)
    keep = rn[["pid", "true", "pred"]].rename(columns={"pred": "pred_resnet", "true": "true_resnet"})
    merged = pred_l0.merge(frame[["pid", "long_cm", "label_5class", "phase"]], on="pid", how="inner")
    if len(merged) != len(frame) or set(merged["pid"].astype(str)) != set(frame["pid"].astype(str)):
        raise SystemExit("L0 predictions do not cover the external cohort")
    merged = merged.merge(keep, on="pid", how="left")
    if merged["pred_resnet"].isna().any():
        missing = merged.loc[merged["pred_resnet"].isna(), "pid"].tolist()
        raise SystemExit(f"saved predictions missing {len(missing)} ids")
    if not (merged["label_5class"] == merged["true"]).all():
        raise SystemExit("L0 true label differs from the feature label")
    if not (merged["label_5class"] == merged["true_resnet"]).all():
        raise SystemExit("saved prediction label differs from the feature label")
    return merged


def fmt4(v):
    if v != v:
        return "—"
    return f"{v:.4f}"


def pm(mean, std):
    return f"{mean:.4f}±{std:.4f}"


def write_web(path: Path, agg, rcc_rows, kits_rows):
    def row_of(rows, subset):
        for row in rows:
            if row["subset"] == subset:
                return row
        raise SystemExit(f"missing subset {subset}")

    rcc = row_of(rcc_rows, "all")
    kits = row_of(kits_rows, "all")
    lines = [
        "# L0 size baseline",
        "",
        "Multinomial logistic regression on log(mask volume) and log(axis-aligned long diameter), "
        "class_weight=balanced, C=1.0. Cross-validation uses the fixed folds; "
        "the standard deviation is the population value (ddof=0). "
        "External scores average the five fold-wise probability vectors. "
        "RCC-AID keeps the first three class probabilities and renormalizes each row (denominator floor 1e-8).",
        "",
        "## Cross-validation (UCSF portal venous, 5 classes)",
        "",
        "| Metric | mean±std |",
        "| --- | --- |",
        f"| balanced accuracy | **{pm(agg['bacc']['mean'], agg['bacc']['std'])}** |",
        f"| accuracy | {pm(agg['acc']['mean'], agg['acc']['std'])} |",
        f"| macro-F1 | {pm(agg['macro_f1']['mean'], agg['macro_f1']['std'])} |",
        f"| AUC (OvR) | {pm(agg['auc_ovr']['mean'], agg['auc_ovr']['std'])} |",
        "",
        "## External test (mean of five fold-wise probabilities)",
        "",
        "| Dataset | n | Task | acc | bacc | macro-F1 | AUC (OvR) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        f"| RCC-AID | {rcc['n']} | 3-class | {fmt4(rcc['acc'])} | **{fmt4(rcc['bacc'])}** | {fmt4(rcc['macro_f1'])} | {fmt4(rcc['auc_ovr'])} |",
        f"| KiTS23 | {kits['n']} | 5-class | {fmt4(kits['acc'])} | **{fmt4(kits['bacc'])}** | {fmt4(kits['macro_f1'])} | {fmt4(kits['auc_ovr'])} |",
        "",
        "## RCC-AID by phase",
        "",
        "| Phase | n | acc | bacc | macro-F1 | AUC (OvR) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for phase in ("arterial", "delayed", "venous"):
        row = row_of(rcc_rows, f"phase:{phase}")
        lines.append(
            f"| {phase} | {row['n']} | {fmt4(row['acc'])} | {fmt4(row['bacc'])} | {fmt4(row['macro_f1'])} | {fmt4(row['auc_ovr'])} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", path, flush=True)


def main():
    args = parse_args()
    refuse_disallowed(
        [
            args.split,
            args.qc_seg,
            args.ucsf_long_cm,
            args.ucsf_ref,
            args.rcc_cases,
            args.rcc_roi,
            args.kits_cases,
            args.kits_roi,
            args.resnet_rcc,
            args.resnet_kits,
            args.out,
        ]
    )
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    ucsf = feature_frame(load_ucsf(args, out))
    rcc = feature_frame(load_rcc(args))
    kits = feature_frame(load_kits(args))
    print(f"n_split={len(ucsf)}", flush=True)
    print(f"n_rccaid={len(rcc)}", flush=True)
    print(f"n_kits23={len(kits)}", flush=True)
    if len(ucsf) != N_SPLIT or len(rcc) != N_RCC or len(kits) != N_KITS:
        raise SystemExit(
            f"count check failed: split={len(ucsf)} rcc={len(rcc)} kits={len(kits)}"
        )
    folds = sorted(int(v) for v in ucsf["fold"].unique())
    if folds != [0, 1, 2, 3, 4]:
        raise SystemExit(f"fold ids {folds}")
    print("scaler_fit=train_fold_only", flush=True)
    print("features=log_volume,log_long C=1.0 class_weight=balanced", flush=True)

    write_features(out / "size_features.csv", [ucsf, rcc, kits])

    models = []
    fold_rows = []
    coef_rows = []
    oof_parts = []
    for fold in range(5):
        train = ucsf[ucsf["fold"].astype(int) != fold]
        valid = ucsf[ucsf["fold"].astype(int) == fold]
        scaler, clf = fit_fold(train)
        models.append((scaler, clf))
        print(f"classes_={[int(c) for c in clf.classes_]}", flush=True)
        prob = proba_of(scaler, clf, valid)
        m = metrics_np(valid["y"].to_numpy(), prob, CLASSES5)
        m["subset"] = f"fold{fold}"
        fold_rows.append(
            {
                "fold": fold,
                "n_val": int(len(valid)),
                "n_train": int(len(train)),
                "acc": m["acc"],
                "bacc": m["bacc"],
                "macro_f1": m["macro_f1"],
                "auc_ovr": m["auc_ovr"],
                "best_bacc": m["bacc"],
                "checkpoint": "",
                "cm_json": json.dumps(m["cm"]),
                **{
                    f"{name}_{metric}": detail[metric]
                    for name, detail in m["per_class"].items()
                    for metric in ("recall", "precision", "f1", "support")
                },
            }
        )
        pred = prob.argmax(1)
        oof = valid[["pid", "fold", "label_5class", "long_cm"]].copy()
        oof["true"] = oof["label_5class"]
        oof["pred"] = [CLASSES5[int(i)] for i in pred]
        for k, name in enumerate(CLASSES5):
            oof[f"p_{name}"] = prob[:, k]
        oof_parts.append(oof)
        for k, name in enumerate(CLASSES5):
            coef_rows.append(
                {
                    "fold": fold,
                    "class": name,
                    "coef_log_volume": float(clf.coef_[k, 0]),
                    "coef_log_long": float(clf.coef_[k, 1]),
                    "intercept": float(clf.intercept_[k]),
                    "n_iter": int(np.max(clf.n_iter_)),
                }
            )
        print(
            f"fold{fold} n_train={len(train)} n_val={len(valid)} bacc={m['bacc']:.6f}",
            flush=True,
        )

    oof = pd.concat(oof_parts, ignore_index=True)
    oof_path = out / "cv_oof_predictions.csv"
    oof.drop(columns=["label_5class"]).to_csv(oof_path, index=False)
    print("wrote", oof_path, flush=True)
    pd.DataFrame(fold_rows).to_csv(out / "cv_fold_metrics.csv", index=False)
    print("wrote", out / "cv_fold_metrics.csv", flush=True)
    pd.DataFrame(coef_rows).to_csv(out / "cv_coefficients.csv", index=False)
    print("wrote", out / "cv_coefficients.csv", flush=True)

    agg = {
        "n_split": N_SPLIT,
        "n_rccaid": N_RCC,
        "n_kits23": N_KITS,
        "classes_": [0, 1, 2, 3, 4],
        "scaler_fit": "train_fold_only",
        "features": FEATS,
        "C": 1.0,
        "class_weight": "balanced",
        "std_ddof": 0,
        "n_models": len(models),
        "acc": mean_std([r["acc"] for r in fold_rows]),
        "bacc": mean_std([r["bacc"] for r in fold_rows]),
        "macro_f1": mean_std([r["macro_f1"] for r in fold_rows]),
        "auc_ovr": mean_std([r["auc_ovr"] for r in fold_rows]),
        "ucsf_long_cm_min": float(ucsf["long_cm"].min()),
        "ucsf_long_cm_max": float(ucsf["long_cm"].max()),
        "ucsf_ref_diameter_n": int(ucsf["ref_diameter_cm"].notna().sum()),
        "kits_ref_diameter_n": int(kits["ref_diameter_cm"].notna().sum()),
        "rcc_ref_diameter_n": 0,
    }
    (out / "cv_aggregate.json").write_text(json.dumps(json_safe(agg), indent=2) + "\n", encoding="utf-8")
    print("wrote", out / "cv_aggregate.json", flush=True)
    print(
        "cv_bacc mean={mean:.6f} std={std:.6f} ddof=0".format(**agg["bacc"]),
        flush=True,
    )

    def eval_external(frame, cohort_prefix_min, phase_key, class_mode):
        prob5 = np.mean([proba_of(s, c, frame) for s, c in models], axis=0)
        if prob5.shape[1] != 5:
            raise SystemExit(f"expected 5-class probabilities, got {prob5.shape}")
        if class_mode == "rcc3":
            prob = prob5[:, :3]
            prob = prob / np.clip(prob.sum(axis=1, keepdims=True), 1e-8, None)
            names = CLASSES3
            prefix = "phase"
            min_n = 1
        else:
            prob = prob5
            names = CLASSES5
            prefix = "phase_est"
            min_n = 2
        y = frame["y"].to_numpy(dtype=int)
        if class_mode == "rcc3":
            y = y  # already 0..2 for RCC labels
        rows = []
        overall = metrics_np(y, prob, names)
        overall["subset"] = "all"
        rows.append(overall)
        by_phase = {}
        phases = frame["phase"].tolist()
        for phase in sorted({p for p in phases if p}):
            idx = np.array([i for i, p in enumerate(phases) if p == phase])
            if len(idx) < min_n:
                continue
            m = metrics_np(y[idx], prob[idx], names)
            m["subset"] = f"{prefix}:{phase}"
            rows.append(m)
            by_phase[phase] = {k: v for k, v in m.items() if k != "y_pred"}
        return prob, names, rows, by_phase

    rcc_prob, rcc_names, rcc_rows, rcc_phase = eval_external(rcc, 1, "phase", "rcc3")
    kits_prob, kits_names, kits_rows, kits_phase = eval_external(kits, 2, "phase_est", "kits5")

    write_metrics_csv(out / "ext_rccaid_metrics.csv", rcc_rows)
    write_metrics_csv(out / "ext_kits23_metrics.csv", kits_rows)
    write_predictions(
        out / "ext_rccaid_metrics_predictions.csv",
        "phase",
        rcc["pid"].tolist(),
        rcc["phase"].tolist(),
        rcc["label_5class"].tolist(),
        rcc_names,
        rcc_prob,
    )
    write_predictions(
        out / "ext_kits23_metrics_predictions.csv",
        "phase_est",
        kits["pid"].tolist(),
        kits["phase"].tolist(),
        kits["label_5class"].tolist(),
        kits_names,
        kits_prob,
    )
    maybe_cm_png(rcc_rows[0]["cm"], rcc_names, out / "ext_rccaid_metrics_cm.png")
    maybe_cm_png(kits_rows[0]["cm"], kits_names, out / "ext_kits23_metrics_cm.png")

    def dump_json(path, dataset, rows, by_phase):
        detail = {
            "dataset": dataset,
            "n_models": 5,
            "ensemble": "probability_mean",
            "n_cases": rows[0]["n"],
            "classes_": [0, 1, 2, 3, 4],
            "scaler_fit": "train_fold_only",
            "overall": {k: v for k, v in rows[0].items() if k != "y_pred"},
            "by_phase": by_phase,
        }
        path.write_text(json.dumps(json_safe(detail), indent=2) + "\n", encoding="utf-8")
        print("wrote", path, flush=True)

    dump_json(out / "ext_rccaid_metrics.json", "rccaid", rcc_rows, rcc_phase)
    dump_json(out / "ext_kits23_metrics.json", "kits23", kits_rows, kits_phase)

    rcc_l0 = pd.read_csv(out / "ext_rccaid_metrics_predictions.csv", dtype=str)
    kits_l0 = pd.read_csv(out / "ext_kits23_metrics_predictions.csv", dtype=str)
    rcc_joined = join_resnet(rcc, rcc_l0, args.resnet_rcc, "phase")
    kits_joined = join_resnet(kits, kits_l0, args.resnet_kits, "phase_est")
    ucsf_max = float(ucsf["long_cm"].max())
    tertile = tertile_table(oof, rcc_joined, kits_joined, args.resnet_rcc, args.resnet_kits, ucsf_max)
    tertile_path = out / "size_tertile_bacc.csv"
    tertile.to_csv(tertile_path, index=False)
    print("wrote", tertile_path, flush=True)
    write_web(out / "web_table_rows.md", agg, rcc_rows, kits_rows)
    print(
        "ext_rccaid bacc={bacc:.6f} acc={acc:.6f} n={n}".format(**rcc_rows[0]),
        flush=True,
    )
    print(
        "ext_kits23 bacc={bacc:.6f} acc={acc:.6f} n={n}".format(**kits_rows[0]),
        flush=True,
    )


if __name__ == "__main__":
    main()
