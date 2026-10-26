#!/usr/bin/env python3
"""Linear probe on the fixed folds. Features are the caller-supplied vectors only; volume and diameter are not added."""

from __future__ import annotations

import json
from pathlib import Path

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
LABEL_MAP = {name: i for i, name in enumerate(CLASSES5)}
N_UCSF = 532
N_RCC = 91
N_KITS = 427


def refuse_disallowed(paths):
    token = "batch" + "1"
    for path in paths:
        if token in str(path).lower():
            raise SystemExit("an input path is outside this protocol")


def feature_columns(frame: pd.DataFrame):
    cols = [c for c in frame.columns if c.startswith("f") and c[1:].isdigit()]
    if not cols:
        raise SystemExit("no feature columns")
    banned = [c for c in cols if any(s in c.lower() for s in ("volume", "diameter", "long"))]
    if banned:
        raise SystemExit(f"banned feature names: {banned}")
    return cols


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
    arr = np.asarray(vals, dtype=float)
    return {"mean": float(np.nanmean(arr)), "std": float(np.nanstd(arr, ddof=0))}


def fmt_num(value):
    if value != value:
        return ""
    return f"{float(value):.6f}"


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


def json_safe(obj):
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, float) and obj != obj:
        return None
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return None if value != value else value
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def _prep_fold(train: pd.DataFrame, columns: list):
    values = train[columns].apply(pd.to_numeric, errors="coerce")
    kept = []
    dropped = []
    for col in columns:
        if int(values[col].notna().sum()) == 0:
            dropped.append(col)
            continue
        kept.append(col)
    if not kept:
        raise SystemExit("no usable feature columns in a training fold")
    median = values[kept].median(axis=0, skipna=True)
    return kept, median, dropped


def _apply_prep(frame, columns, median):
    values = frame[columns].apply(pd.to_numeric, errors="coerce")
    return values.fillna(median)


def _fit_fold(x_train, y_train):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(x_train)
    clf = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=4000,
        random_state=20261024,
    )
    clf.fit(scaled, y_train)
    got = [int(c) for c in clf.classes_]
    if got != [0, 1, 2, 3, 4]:
        raise SystemExit(f"class order {got}")
    return scaler, clf


def _predict(scaler, clf, frame, columns, median):
    scaled = scaler.transform(_apply_prep(frame, columns, median).to_numpy(dtype=float))
    return clf.predict_proba(scaled)


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
    for phase in sorted({p for p in phases if p and p == p}):
        idx = np.array([i for i, item in enumerate(phases) if item == phase])
        if len(idx) < min_n:
            continue
        row = metrics_np(y[idx], prob[idx], names)
        row["subset"] = f"{prefix}:{phase}"
        rows.append(row)
        by_phase[phase] = row
    return prob, names, rows, by_phase


def fmt4(value):
    if value != value:
        return "—"
    return f"{float(value):.4f}"


def pm(stat):
    return f"{stat['mean']:.4f}±{stat['std']:.4f}"


def write_web(path: Path, fm: str, agg, rcc_rows, kits_rows):
    def row_of(rows, subset):
        for row in rows:
            if row["subset"] == subset:
                return row
        raise SystemExit(f"missing subset {subset}")

    rcc = row_of(rcc_rows, "all")
    kits = row_of(kits_rows, "all")
    lines = [
        f"# L4 {fm} linear probe",
        "",
        "Frozen foundation-model features and a logistic-regression head. C=1.0, class_weight=balanced. "
        "Median imputation and StandardScaler are fit on each training fold only. "
        "Cross-validation uses the fixed folds; the standard deviation is the population value (ddof=0). "
        "External scores average the five fold-wise probability vectors. "
        "RCC-AID keeps the first three class probabilities and renormalizes each row (denominator floor 1e-8). "
        "KiTS23 keeps five classes. Features do not include volume or diameter.",
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
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", path, flush=True)


def write_predictions(path, phase_col, pids, phases, labels, names, prob):
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    pred = prob.argmax(1)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pid", phase_col, "true", "pred"] + [f"p_{c}" for c in names])
        for i, pid in enumerate(pids):
            writer.writerow(
                [pid, phases[i], labels[i], names[int(pred[i])]]
                + [f"{float(prob[i, k]):.6f}" for k in range(len(names))]
            )
    print("wrote", path, flush=True)


def run_probe(feature_dir: Path, out_dir: Path, fm: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for cohort, filename in (
        ("ucsf", f"features_{fm}_ucsf.csv"),
        ("rccaid", f"features_{fm}_rccaid.csv"),
        ("kits23", f"features_{fm}_kits23.csv"),
    ):
        path = feature_dir / filename
        if not path.is_file():
            raise SystemExit(f"missing features {path}")
        frame = pd.read_csv(path, dtype={"pid": str, "label": str, "phase": str, "fold": str})
        frame["cohort"] = cohort
        parts.append(frame)
    data = pd.concat(parts, ignore_index=True)
    if data["pid"].duplicated().any():
        raise SystemExit("duplicate feature pid")
    cols = feature_columns(data)
    ucsf = data[data["cohort"] == "ucsf"].copy()
    rcc = data[data["cohort"] == "rccaid"].copy()
    kits = data[data["cohort"] == "kits23"].copy()
    if len(ucsf) != N_UCSF or len(rcc) != N_RCC or len(kits) != N_KITS:
        raise SystemExit(f"row count ucsf={len(ucsf)} rcc={len(rcc)} kits={len(kits)}")
    for frame in (ucsf, rcc, kits):
        mapped = frame["label"].map(LABEL_MAP)
        if mapped.isna().any():
            raise SystemExit("label outside the five classes")
        frame["y"] = mapped.astype(int)
    if set(rcc["y"].tolist()) - {0, 1, 2}:
        raise SystemExit("RCC-AID label outside three classes")
    ucsf["fold_id"] = ucsf["fold"].astype(int)
    folds = sorted(int(v) for v in ucsf["fold_id"].unique())
    if folds != [0, 1, 2, 3, 4]:
        raise SystemExit(f"fold ids {folds}")

    models = []
    fold_rows = []
    oof_parts = []
    prep_log = {}
    for fold in range(5):
        train = ucsf[ucsf["fold_id"] != fold]
        valid = ucsf[ucsf["fold_id"] == fold]
        kept, median, dropped = _prep_fold(train, cols)
        x_train = _apply_prep(train, kept, median).to_numpy(dtype=float)
        scaler, clf = _fit_fold(x_train, train["y"].to_numpy(dtype=int))
        models.append((scaler, clf, kept, median))
        prob = _predict(scaler, clf, valid, kept, median)
        row = metrics_np(valid["y"].to_numpy(dtype=int), prob, CLASSES5)
        fold_rows.append({
            "fold": fold,
            "n_train": int(len(train)),
            "n_val": int(len(valid)),
            "n_features": len(kept),
            "n_dropped_all_nan": len(dropped),
            "acc": row["acc"],
            "bacc": row["bacc"],
            "macro_f1": row["macro_f1"],
            "auc_ovr": row["auc_ovr"],
            "cm_json": json.dumps(row["cm"]),
        })
        part = valid[["pid", "fold"]].copy()
        part["true"] = valid["label"].to_numpy()
        part["pred"] = [CLASSES5[int(i)] for i in prob.argmax(1)]
        for k, name in enumerate(CLASSES5):
            part[f"p_{name}"] = prob[:, k]
        oof_parts.append(part)
        prep_log[f"fold{fold}"] = {"n_features": len(kept), "dropped_all_nan": dropped}
        print(f"fold{fold} bacc={row['bacc']:.6f} n_features={len(kept)}", flush=True)

    oof = pd.concat(oof_parts, ignore_index=True)
    oof.to_csv(out_dir / "cv_oof_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(out_dir / "cv_fold_metrics.csv", index=False)
    agg = {
        "n_split": N_UCSF,
        "n_rccaid": N_RCC,
        "n_kits23": N_KITS,
        "ddof": 0,
        "C": 1.0,
        "class_weight": "balanced",
        "scaler": "StandardScaler",
        "imputer": "train_fold_median",
        "n_features": len(cols),
        "acc": mean_std([r["acc"] for r in fold_rows]),
        "bacc": mean_std([r["bacc"] for r in fold_rows]),
        "macro_f1": mean_std([r["macro_f1"] for r in fold_rows]),
        "auc_ovr": mean_std([r["auc_ovr"] for r in fold_rows]),
        "folds": prep_log,
    }
    (out_dir / "cv_aggregate.json").write_text(json.dumps(json_safe(agg), indent=2) + "\n", encoding="utf-8")

    def ensemble(frame):
        acc = None
        for scaler, clf, kept, median in models:
            prob = _predict(scaler, clf, frame, kept, median)
            acc = prob if acc is None else acc + prob
        return acc / float(len(models))

    rcc_prob5 = ensemble(rcc)
    kits_prob = ensemble(kits)
    rcc_prob, rcc_names, rcc_rows, rcc_phase = eval_external(
        rcc_prob5, rcc["y"].to_numpy(dtype=int), rcc["phase"].tolist(), "rcc3"
    )
    _, kits_names, kits_rows, kits_phase = eval_external(
        kits_prob, kits["y"].to_numpy(dtype=int), kits["phase"].tolist(), "kits5"
    )
    write_metrics_csv(out_dir / "ext_rccaid_metrics.csv", rcc_rows)
    write_metrics_csv(out_dir / "ext_kits23_metrics.csv", kits_rows)
    write_predictions(
        out_dir / "ext_rccaid_predictions.csv",
        "phase",
        rcc["pid"].tolist(),
        rcc["phase"].tolist(),
        rcc["label"].tolist(),
        rcc_names,
        rcc_prob,
    )
    write_predictions(
        out_dir / "ext_kits23_predictions.csv",
        "phase_est",
        kits["pid"].tolist(),
        kits["phase"].tolist(),
        kits["label"].tolist(),
        kits_names,
        kits_prob,
    )
    write_web(out_dir / "web_table_rows.md", fm, agg, rcc_rows, kits_rows)
    summary = {
        "bacc_cv": agg["bacc"],
        "bacc_rccaid": rcc_rows[0]["bacc"],
        "bacc_kits23": kits_rows[0]["bacc"],
        "n_rcc_phase": {k: v["n"] for k, v in rcc_phase.items()},
        "n_kits_phase": {k: v["n"] for k, v in kits_phase.items()},
    }
    (out_dir / "probe_summary.json").write_text(json.dumps(json_safe(summary), indent=2) + "\n", encoding="utf-8")
    print("probe_done", json.dumps(json_safe(summary)), flush=True)
    return summary
