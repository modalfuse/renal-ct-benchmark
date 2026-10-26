#!/usr/bin/env python3
"""Append one feature row per case and log cases that could not be extracted."""

from __future__ import annotations

import csv
from pathlib import Path


class FeatureStore:
    def __init__(self, feature_dir: Path, fm: str, dim: int):
        self.feature_dir = feature_dir
        self.fm = fm
        self.dim = dim
        self.paths = {
            "ucsf": feature_dir / f"features_{fm}_ucsf.csv",
            "rccaid": feature_dir / f"features_{fm}_rccaid.csv",
            "kits23": feature_dir / f"features_{fm}_kits23.csv",
        }
        feature_dir.mkdir(parents=True, exist_ok=True)
        self.header = ["pid", "cohort", "label", "phase", "fold"] + [f"f{i:04d}" for i in range(dim)]
        self.done = {}
        for cohort, path in self.paths.items():
            self.done[cohort] = set()
            if not path.is_file():
                continue
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.reader(handle)
                header = next(reader, None)
                if header is None:
                    continue
                feats = [c for c in header if c.startswith("f") and c[1:].isdigit()]
                if len(feats) != dim:
                    raise SystemExit(f"{path} dim {len(feats)} != {dim}")
                for row in reader:
                    if row:
                        self.done[cohort].add(row[0])

    def has(self, cohort: str, pid: str) -> bool:
        return pid in self.done[cohort]

    def write(self, case: dict, vector):
        if len(vector) != self.dim:
            raise SystemExit(f"vector dim {len(vector)} != {self.dim}")
        path = self.paths[case["cohort"]]
        new_file = not path.is_file() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if new_file:
                writer.writerow(self.header)
            writer.writerow(
                [case["pid"], case["cohort"], case["label"], case["phase"], case["fold"]]
                + [f"{float(v):.8g}" for v in vector]
            )
            handle.flush()
        self.done[case["cohort"]].add(case["pid"])


class ErrorLog:
    def __init__(self, path: Path):
        self.path = path
        self.rows = []
        path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, pid: str, cohort: str, reason: str, detail: str):
        row = {"pid": pid, "cohort": cohort, "reason": reason, "detail": detail}
        self.rows.append(row)
        new_file = not self.path.is_file() or self.path.stat().st_size == 0
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["pid", "cohort", "reason", "detail"])
            if new_file:
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
        print(f"ERROR {cohort} {pid} {reason} {detail}", flush=True)
