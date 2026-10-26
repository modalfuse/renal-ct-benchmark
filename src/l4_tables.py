#!/usr/bin/env python3
"""Read the fixed fold file, QC table, and external-task table."""

from __future__ import annotations

import csv
import os
from pathlib import Path

from l4_probe import N_KITS, N_RCC, N_UCSF, refuse_disallowed


def _required(name: str) -> Path:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"set {name}")
    return Path(value)


def read_cases():
    split = _required("L4_SPLIT")
    qc_path = _required("L4_QC")
    ext = _required("L4_EXT")
    roi = _required("L4_ROI")
    refuse_disallowed([split, qc_path, ext, roi])
    with split.open(newline="", encoding="utf-8") as handle:
        split_rows = list(csv.DictReader(handle))
    if len(split_rows) != N_UCSF:
        raise SystemExit(f"split rows {len(split_rows)}")
    folds = sorted({int(row["fold"]) for row in split_rows})
    if folds != [0, 1, 2, 3, 4]:
        raise SystemExit(f"fold ids {folds}")
    with qc_path.open(newline="", encoding="utf-8") as handle:
        qc = {row["pid"]: row for row in csv.DictReader(handle)}
    with ext.open(newline="", encoding="utf-8") as handle:
        ext_rows = list(csv.DictReader(handle))
    n_rcc = sum(row["cohort"] == "rccaid" for row in ext_rows)
    n_kits = sum(row["cohort"] == "kits23" for row in ext_rows)
    if n_rcc != N_RCC or n_kits != N_KITS or len(ext_rows) != N_RCC + N_KITS:
        raise SystemExit(f"external rows rcc={n_rcc} kits={n_kits} total={len(ext_rows)}")
    ucsf = []
    for row in split_rows:
        pid = row["pid"]
        meta = qc.get(pid)
        ucsf.append({
            "pid": pid,
            "cohort": "ucsf",
            "label": row["label_5class"],
            "phase": "portven",
            "fold": row["fold"],
            "h5": "" if meta is None else meta.get("input", ""),
            "spacing": "" if meta is None else meta.get("spacing", ""),
            "img": str(roi / f"{pid}_img.nii.gz"),
            "mask": str(roi / f"{pid}_mask.nii.gz"),
            "qc_label": "" if meta is None else meta.get("label_5class", ""),
        })
    external = []
    for row in ext_rows:
        external.append({
            "pid": row["pid"],
            "cohort": row["cohort"],
            "label": row["label"],
            "phase": row.get("phase", ""),
            "fold": "",
            "img": row["img"],
            "mask": row["mask"],
        })
    return ucsf, external
