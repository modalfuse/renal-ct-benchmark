#!/usr/bin/env python3
"""L2 radiomics baseline: PyRadiomics original features and a frozen XGBoost model.

External cohorts are extracted here. UCSF features are read from a CSV produced
on the read-only host. Per-fold preprocessing is fit on that fold's training
rows only. External probabilities are the mean of the five fold models.
"""

from __future__ import annotations

import argparse
import json
import logging
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from xgboost import XGBClassifier

CLASSES5 = ["ccRCC", "pRCC", "chRCC", "oncocytoma", "AML"]
CLASSES3 = ["ccRCC", "pRCC", "chRCC"]
FEATURE_CLASSES = ("shape", "firstorder", "glcm", "glrlm", "glszm", "ngtdm", "gldm")
N_SPLIT = 532
N_RCC = 91
N_KITS = 427
BANNED = {
    "volume_ml",
    "long_cm",
    "log_volume",
    "log_long",
    "ref_diameter_cm",
    "tumor_size_cm",
    "radiographic_size_cm",
    "pathologic_size_cm",
}


def parse_args():
    p = argparse.ArgumentParser(description="L2 radiomics XGBoost baseline")
    p.add_argument("--fold", required=True, type=Path)
    p.add_argument("--ucsf-features", required=True, type=Path)
    p.add_argument("--ucsf-errors", type=Path, default=None)
    p.add_argument("--features-ext", type=Path, default=None)
    p.add_argument("--rcc-cases", type=Path, default=None)
    p.add_argument("--rcc-roi", type=Path, default=None)
    p.add_argument("--kits-cases", type=Path, default=None)
    p.add_argument("--kits-roi", type=Path, default=None)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--workers", type=int, default=8)
    return p.parse_args()


def refuse_disallowed(paths):
    token = "batch" + "1"
    for path in paths:
        if token in str(path).lower():
            raise SystemExit("an input path is outside this protocol")


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


def extract_one(task):
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
                "errors": [{
                    "reason": "grid_mismatch",
                    "detail": f"img {image.shape} {z_img} mask {mask.shape} {z_msk}",
                }],
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
        return {"pid": pid, "row": row, "errors": errors}
    except Exception as exc:
        errors.append({"reason": "extract_failed", "detail": f"{type(exc).__name__}: {exc}"})
        return {"pid": pid, "errors": errors}


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


def mean_std(vals):
    a = np.asarray(vals, dtype=float)
    return {"mean": float(np.nanmean(a)), "std": float(np.nanstd(a, ddof=0))}


def fmt_num(v):
    if v != v:
        return ""
    return f"{float(v):.6f}"


def write_metrics_csv(path: Path, rows: list):
    import csv

    fields = ["subset", "n", "acc", "bacc", "macro_f1", "auc_ovr", "cm_json"]
    for name in CLASSES5:
        for metric in ("recall", "precision", "f1", "support"):
            fields.append(f"{name}_{metric}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
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


def write_predictions(path: Path, phase_col, pids, phases, labels, names, prob):
    import csv

    pred = prob.argmax(1)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pid", phase_col, "true", "pred"] + [f"p_{c}" for c in names])
        for i, pid in enumerate(pids):
            writer.writerow(
                [pid, phases[i], labels[i], names[int(pred[i])]]
                + [f"{prob[i, k]:.6f}" for k in range(len(names))]
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


def feature_columns(frame: pd.DataFrame):
    cols = [c for c in frame.columns if c.startswith("original_")]
    banned = sorted(set(cols) & BANNED)
    if banned:
        raise SystemExit(f"banned columns in model features: {banned}")
    if any(not c.startswith("original_") for c in cols):
        raise SystemExit("model features must be original_*")
    return cols


def prepare_fold(train: pd.DataFrame, columns: list):
    x = train[columns]
    nan_frac = x.isna().mean()
    drop_nan = [c for c in columns if float(nan_frac[c]) > 0.5]
    drop_var = [c for c in columns if int(x[c].nunique(dropna=True)) <= 1]
    dropped = set(drop_nan) | set(drop_var)
    kept = [c for c in columns if c not in dropped]
    med = x[kept].median(numeric_only=True)
    filled = [c for c in kept if bool(x[c].isna().any())]
    record = {
        "dropped_nan_gt_50": drop_nan,
        "dropped_zero_variance": drop_var,
        "median_filled": filled,
        "n_features": len(kept),
        "columns": kept,
    }
    return med, record


def apply_prep(frame, columns, median):
    out = frame.reindex(columns=columns)
    return out.fillna(median)


def balanced_sample_weight(y):
    y = np.asarray(y, dtype=int)
    n = len(y)
    weight = np.empty(n, dtype=float)
    for cls in range(5):
        n_i = int(np.sum(y == cls))
        if n_i == 0:
            raise SystemExit(f"class {cls} absent from a training fold")
        weight[y == cls] = n / (5.0 * n_i)
    return weight


def fit_fold(x_train, y_train):
    if x_train.shape[1] < 1:
        raise SystemExit("no features left after preprocessing")
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=5,
        n_estimators=300,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0,
        min_child_weight=1,
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=20261024,
        n_jobs=-1,
    )
    model.fit(x_train, y_train, sample_weight=balanced_sample_weight(y_train))
    got = [int(c) for c in model.classes_]
    if got != [0, 1, 2, 3, 4]:
        raise SystemExit(f"class order {got}")
    return model


def run_extract(tasks, workers):
    ctx = get_context("spawn")
    with ctx.Pool(workers) as pool:
        return list(pool.imap_unordered(extract_one, tasks, chunksize=1))


def collect(results, cohort):
    rows, errors = [], []
    for item in results:
        for err in item.get("errors") or []:
            errors.append({"pid": item["pid"], "cohort": cohort, **err})
        if "row" in item:
            rows.append(item["row"])
    return rows, errors


def eval_external(prob, y, phases, mode):
    if mode == "rcc3":
        prob = prob[:, :3]
        prob = prob / np.clip(prob.sum(axis=1, keepdims=True), 1e-8, None)
        names = CLASSES3
        prefix = "phase"
        min_n = 1
    else:
        names = CLASSES5
        prefix = "phase_est"
        min_n = 2
    rows = []
    overall = metrics_np(y, prob, names)
    overall["subset"] = "all"
    rows.append(overall)
    by_phase = {}
    for phase in sorted({p for p in phases if p}):
        idx = np.array([i for i, p in enumerate(phases) if p == phase])
        if len(idx) < min_n:
            continue
        m = metrics_np(y[idx], prob[idx], names)
        m["subset"] = f"{prefix}:{phase}"
        rows.append(m)
        by_phase[phase] = m
    return prob, names, rows, by_phase


def fmt4(v):
    if v != v:
        return "—"
    return f"{v:.4f}"


def pm(stat):
    return f"{stat['mean']:.4f}±{stat['std']:.4f}"


def write_web(path, agg, rcc_rows, kits_rows):
    def row_of(rows, subset):
        for row in rows:
            if row["subset"] == subset:
                return row
        raise SystemExit(f"missing subset {subset}")

    rcc = row_of(rcc_rows, "all")
    kits = row_of(kits_rows, "all")
    lines = [
        "# L2 radiomics baseline",
        "",
        "PyRadiomics original features (shape, firstorder, glcm, glrlm, glszm, ngtdm, gldm; no filters) "
        "and XGBoost. binWidth=25 HU, no resampling, no intensity normalization. "
        "Hyperparameters: n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.8, "
        "colsample_bytree=0.8, balanced sample weights. "
        "Cross-validation uses the fixed folds; the standard deviation is the population value (ddof=0). "
        "External scores average the five fold-wise probability vectors. "
        "RCC-AID keeps the first three class probabilities and renormalizes each row (denominator floor 1e-8).",
        "",
        "## Cross-validation (UCSF portal venous, 5 classes)",
        "",
        "| Metric | mean±std |",
        "| --- | --- |",
        f"| balanced accuracy | **{pm(agg['bacc'])}** |",
        f"| accuracy | {pm(agg['acc'])} |",
        f"| macro-F1 | {pm(agg['macro_f1'])} |",
        f"| AUC (OvR) | {pm(agg['auc_ovr'])} |",
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
            args.fold,
            args.ucsf_features,
            args.ucsf_errors,
            args.rcc_cases,
            args.rcc_roi,
            args.kits_cases,
            args.kits_roi,
            args.out,
        ]
    )
    import xgboost

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    print(f"xgboost={xgboost.__version__}", flush=True)

    split = pd.read_csv(args.fold, dtype={"pid": str, "label_5class": str})
    if list(split.columns[:3]) != ["pid", "label_5class", "fold"]:
        raise SystemExit(f"unexpected fold columns {list(split.columns)}")
    ucsf = pd.read_csv(args.ucsf_features, dtype={"pid": str})
    if len(ucsf) != len(ucsf["pid"].unique()):
        raise SystemExit("duplicate UCSF feature rows")
    missing = sorted(set(split["pid"]) - set(ucsf["pid"]))
    extra = sorted(set(ucsf["pid"]) - set(split["pid"]))
    if missing or extra:
        (out / "missing_ucsf.txt").write_text(
            "missing\n" + "\n".join(missing) + "\nextra\n" + "\n".join(extra) + "\n",
            encoding="utf-8",
        )
        raise SystemExit(f"UCSF id mismatch missing={len(missing)} extra={len(extra)}")
    ucsf = split.merge(ucsf, on="pid", how="inner")
    ucsf["y"] = ucsf["label_5class"].map({n: i for i, n in enumerate(CLASSES5)}).astype(int)
    ucsf["cohort"] = "ucsf"
    ucsf["phase"] = "portven"

    print(f"n_split={len(ucsf)}", flush=True)
    if len(ucsf) != N_SPLIT:
        raise SystemExit("count check failed for UCSF features")
    if args.features_ext is None:
        if not args.rcc_cases or not args.rcc_roi or not args.kits_cases or not args.kits_roi:
            raise SystemExit("external ROI arguments are required when --features-ext is omitted")
        import radiomics
        import SimpleITK as sitk

        print(f"pyradiomics={radiomics.__version__}", flush=True)
        print(f"simpleitk={sitk.Version_VersionString()}", flush=True)
        rcc_cases = pd.read_csv(args.rcc_cases, dtype=str)
        flag = rcc_cases["selected_external"].astype(str).str.lower().isin(["true", "1"])
        rcc_sel = rcc_cases.loc[flag].copy()
        kits_cases = pd.read_csv(args.kits_cases, dtype=str)
        kits_ok = kits_cases[kits_cases["status"].astype(str) == "ok"].copy()
        print(f"n_rccaid={len(rcc_sel)}", flush=True)
        print(f"n_kits23={len(kits_ok)}", flush=True)
        if len(rcc_sel) != N_RCC or len(kits_ok) != N_KITS:
            raise SystemExit("count check failed before extraction")
        tasks = []
        for rec in rcc_sel.itertuples(index=False):
            pid = str(rec.file)
            tasks.append({
                "pid": pid,
                "cohort": "rccaid",
                "label": str(rec.subtype),
                "phase": str(rec.phase),
                "img": str(args.rcc_roi / f"{pid}_img.nii.gz"),
                "mask": str(args.rcc_roi / f"{pid}_mask.nii.gz"),
            })
        for rec in kits_ok.itertuples(index=False):
            pid = str(rec.pid)
            phase = "" if pd.isna(rec.phase_est) else str(rec.phase_est)
            tasks.append({
                "pid": pid,
                "cohort": "kits23",
                "label": str(rec.label_5class),
                "phase": phase,
                "img": str(args.kits_roi / f"{pid}_img.nii.gz"),
                "mask": str(args.kits_roi / f"{pid}_mask.nii.gz"),
            })
        extracted = run_extract(tasks, args.workers)
        by_pid = {item["pid"]: item for item in extracted}
        rcc_rows, rcc_err = collect([by_pid[t["pid"]] for t in tasks if t["cohort"] == "rccaid"], "rccaid")
        kits_rows_f, kits_err = collect([by_pid[t["pid"]] for t in tasks if t["cohort"] == "kits23"], "kits23")
        ext = pd.DataFrame(rcc_rows + kits_rows_f)
        ext.to_csv(out / "features_ext.csv", index=False)
        print("wrote", out / "features_ext.csv", flush=True)
        err_parts = [pd.DataFrame(rcc_err + kits_err)]
        if args.ucsf_errors and args.ucsf_errors.is_file():
            err_parts.append(pd.read_csv(args.ucsf_errors, dtype=str))
        errors = pd.concat(err_parts, ignore_index=True) if any(len(p) for p in err_parts) else pd.DataFrame()
        if len(errors) == 0:
            errors = pd.DataFrame(columns=["pid", "cohort", "reason", "detail"])
        errors.to_csv(out / "feature_errors.csv", index=False)
        print("wrote", out / "feature_errors.csv", flush=True)
        hard = errors[errors["reason"].astype(str) != "hu_range"] if len(errors) else errors
        if len(rcc_rows) != N_RCC or len(kits_rows_f) != N_KITS or len(hard):
            raise SystemExit(
                f"incomplete external extraction rcc={len(rcc_rows)} kits={len(kits_rows_f)} hard={len(hard)}"
            )
        rad_version = radiomics.__version__
        sitk_version = sitk.Version_VersionString()
    else:
        ext = pd.read_csv(args.features_ext, dtype={"pid": str})
        ext.to_csv(out / "features_ext.csv", index=False)
        print(f"n_rccaid={(ext.cohort == 'rccaid').sum()}", flush=True)
        print(f"n_kits23={(ext.cohort == 'kits23').sum()}", flush=True)
        if args.ucsf_errors and args.ucsf_errors.is_file():
            pd.read_csv(args.ucsf_errors, dtype=str).to_csv(out / "feature_errors.csv", index=False)
        else:
            pd.DataFrame(columns=["pid", "cohort", "reason", "detail"]).to_csv(
                out / "feature_errors.csv", index=False
            )
        rad_version = str(ext["pyradiomics_version"].iloc[0]) if "pyradiomics_version" in ext.columns else ""
        sitk_version = str(ext["simpleitk_version"].iloc[0]) if "simpleitk_version" in ext.columns else ""
        print(f"pyradiomics={rad_version}", flush=True)
        print(f"simpleitk={sitk_version}", flush=True)

    cols = feature_columns(ucsf)
    ext_cols = feature_columns(ext)
    if cols != ext_cols and set(cols) != set(ext_cols):
        only_u = sorted(set(cols) - set(ext_cols))
        only_e = sorted(set(ext_cols) - set(cols))
        raise SystemExit(f"feature set mismatch ucsf_only={only_u[:8]} ext_only={only_e[:8]}")
    cols = sorted(set(cols) & set(ext_cols))
    print(f"n_feature_cols={len(cols)}", flush=True)
    print("scaler_fit=not_used tree_preprocess=train_fold_only", flush=True)

    label_map = {n: i for i, n in enumerate(CLASSES5)}
    ext["y"] = ext["label"].map(label_map)
    if ext["y"].isna().any():
        raise SystemExit("external label outside the five classes")
    ext["y"] = ext["y"].astype(int)
    rcc = ext[ext["cohort"] == "rccaid"].reset_index(drop=True)
    kits = ext[ext["cohort"] == "kits23"].reset_index(drop=True)
    if set(rcc["label"]) - set(CLASSES3):
        raise SystemExit("RCC-AID label outside three classes")

    folds = sorted(int(v) for v in ucsf["fold"].unique())
    if folds != [0, 1, 2, 3, 4]:
        raise SystemExit(f"fold ids {folds}")

    models = []
    prep = []
    fold_rows = []
    oof_parts = []
    col_log = {}
    for fold in range(5):
        train = ucsf[ucsf["fold"].astype(int) != fold]
        valid = ucsf[ucsf["fold"].astype(int) == fold]
        median, record = prepare_fold(train, cols)
        col_log[f"fold{fold}"] = record
        x_train = apply_prep(train, record["columns"], median)
        x_valid = apply_prep(valid, record["columns"], median)
        model = fit_fold(x_train.to_numpy(dtype=float), train["y"].to_numpy(dtype=int))
        models.append(model)
        prep.append((record["columns"], median))
        print(f"classes_={[int(c) for c in model.classes_]}", flush=True)
        prob = model.predict_proba(x_valid.to_numpy(dtype=float))
        m = metrics_np(valid["y"].to_numpy(dtype=int), prob, CLASSES5)
        fold_rows.append({
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
            "n_features": record["n_features"],
            "n_dropped_nan": len(record["dropped_nan_gt_50"]),
            "n_dropped_var": len(record["dropped_zero_variance"]),
            "n_median_filled": len(record["median_filled"]),
            **{
                f"{name}_{metric}": detail[metric]
                for name, detail in m["per_class"].items()
                for metric in ("recall", "precision", "f1", "support")
            },
        })
        pred = prob.argmax(1)
        part = valid[["pid", "fold"]].copy()
        part["true"] = valid["label_5class"].to_numpy()
        part["pred"] = [CLASSES5[int(i)] for i in pred]
        for k, name in enumerate(CLASSES5):
            part[f"p_{name}"] = prob[:, k]
        oof_parts.append(part)
        print(
            f"fold{fold} n_train={len(train)} n_val={len(valid)} "
            f"n_features={record['n_features']} bacc={m['bacc']:.6f}",
            flush=True,
        )

    oof = pd.concat(oof_parts, ignore_index=True)
    oof.to_csv(out / "cv_oof_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(out / "cv_fold_metrics.csv", index=False)
    (out / "cv_feature_cols.json").write_text(json.dumps(col_log, indent=2) + "\n", encoding="utf-8")
    print("wrote", out / "cv_feature_cols.json", flush=True)

    def ensemble(frame):
        acc = None
        for model, (columns, median) in zip(models, prep):
            x = apply_prep(frame, columns, median).to_numpy(dtype=float)
            p = model.predict_proba(x)
            acc = p if acc is None else acc + p
        return acc / float(len(models))

    rcc_prob5 = ensemble(rcc)
    kits_prob = ensemble(kits)
    if rcc_prob5.shape[1] != 5 or kits_prob.shape[1] != 5:
        raise SystemExit("expected 5-class probabilities")
    rcc_prob, rcc_names, rcc_metrics, rcc_phase = eval_external(
        rcc_prob5, rcc["y"].to_numpy(dtype=int), rcc["phase"].tolist(), "rcc3"
    )
    _, kits_names, kits_metrics, kits_phase = eval_external(
        kits_prob, kits["y"].to_numpy(dtype=int), kits["phase"].tolist(), "kits5"
    )
    write_metrics_csv(out / "ext_rccaid_metrics.csv", rcc_metrics)
    write_metrics_csv(out / "ext_kits23_metrics.csv", kits_metrics)
    write_predictions(
        out / "ext_rccaid_metrics_predictions.csv",
        "phase",
        rcc["pid"].tolist(),
        rcc["phase"].tolist(),
        rcc["label"].tolist(),
        rcc_names,
        rcc_prob,
    )
    write_predictions(
        out / "ext_kits23_metrics_predictions.csv",
        "phase_est",
        kits["pid"].tolist(),
        kits["phase"].tolist(),
        kits["label"].tolist(),
        kits_names,
        kits_prob,
    )
    maybe_cm_png(rcc_metrics[0]["cm"], rcc_names, out / "ext_rccaid_metrics_cm.png")
    maybe_cm_png(kits_metrics[0]["cm"], kits_names, out / "ext_kits23_metrics_cm.png")

    def dump(path, dataset, rows, by_phase):
        detail = {
            "dataset": dataset,
            "n_models": 5,
            "ensemble": "probability_mean",
            "n_cases": rows[0]["n"],
            "classes_": [0, 1, 2, 3, 4],
            "preprocess": "train_fold_only",
            "pyradiomics": rad_version,
            "simpleitk": sitk_version,
            "xgboost": xgboost.__version__,
            "n_feature_cols": len(cols),
            "overall": rows[0],
            "by_phase": by_phase,
        }
        path.write_text(json.dumps(json_safe(detail), indent=2) + "\n", encoding="utf-8")
        print("wrote", path, flush=True)

    dump(out / "ext_rccaid_metrics.json", "rccaid", rcc_metrics, rcc_phase)
    dump(out / "ext_kits23_metrics.json", "kits23", kits_metrics, kits_phase)
    agg = {
        "n_split": N_SPLIT,
        "n_rccaid": N_RCC,
        "n_kits23": N_KITS,
        "classes_": [0, 1, 2, 3, 4],
        "preprocess": "train_fold_only",
        "n_feature_cols": len(cols),
        "pyradiomics": rad_version,
        "simpleitk": sitk_version,
        "xgboost": xgboost.__version__,
        "std_ddof": 0,
        "n_models": 5,
        "acc": mean_std([r["acc"] for r in fold_rows]),
        "bacc": mean_std([r["bacc"] for r in fold_rows]),
        "macro_f1": mean_std([r["macro_f1"] for r in fold_rows]),
        "auc_ovr": mean_std([r["auc_ovr"] for r in fold_rows]),
        "fold_n_features": [r["n_features"] for r in fold_rows],
        "fold_n_dropped_nan": [r["n_dropped_nan"] for r in fold_rows],
        "fold_n_dropped_var": [r["n_dropped_var"] for r in fold_rows],
        "fold_n_median_filled": [r["n_median_filled"] for r in fold_rows],
    }
    (out / "cv_aggregate.json").write_text(json.dumps(json_safe(agg), indent=2) + "\n", encoding="utf-8")
    print("wrote", out / "cv_aggregate.json", flush=True)
    write_web(out / "web_table_rows.md", agg, rcc_metrics, kits_metrics)
    print("cv_bacc mean={mean:.6f} std={std:.6f} ddof=0".format(**agg["bacc"]), flush=True)
    print("ext_rccaid bacc={bacc:.6f} n={n}".format(**rcc_metrics[0]), flush=True)
    print("ext_kits23 bacc={bacc:.6f} n={n}".format(**kits_metrics[0]), flush=True)


if __name__ == "__main__":
    main()
