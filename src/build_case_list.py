#!/usr/bin/env python3
"""Build meta/case_list_all.csv from the cohort tables in --meta-dir.

Each row has the five-class label, whether a portal-venous volume exists,
usability, the chosen box path (relative to meta/), and its source.
Box priority: usable-table npz_path, then pv_boxes_rescue, then pv_boxes_rescue_pending, then pv_boxes.

Usage: python build_case_list.py --meta-dir /path/to/meta
"""
import argparse, csv, json, os
ap = argparse.ArgumentParser(); ap.add_argument("--meta-dir", required=True); a = ap.parse_args()
M = a.meta_dir
def rd(n):
    with open(os.path.join(M, n), encoding="utf-8-sig") as f: return {r["pid"]: r for r in csv.DictReader(f)}
man = rd("ucsf_rmac_manifest.csv"); lab = rd("ucsf_rmac_labels_5class.csv"); use = rd("ucsf_rmac_pv_usable_v2 20261020.csv")
pend16 = {c["pid"] for c in json.load(open(os.path.join(M, "visual_decision_16_cases.json"), encoding="utf-8"))}
rows = []
for pid in sorted(man):
    m = man[pid]; l = lab.get(pid, {}); u = use.get(pid, {})
    cand = []
    if u.get("npz_path"): cand.append((u["npz_path"], "usable_v2:" + (u.get("box_source") or "")))
    cand += [(f"pv_boxes_rescue/{pid}.npz", "rescue"), (f"pv_boxes_rescue_pending/{pid}.npz", "rescue_pending"), (f"pv_boxes/{pid}.npz", "original")]
    box, src = "", ""
    for p, s in cand:
        if os.path.exists(os.path.join(M, p)): box, src = p, s; break
    rows.append(dict(pid=pid, label_5class=l.get("label_5class", ""), include_main=l.get("include_main", ""), bilateral=l.get("bilateral", ""),
                     has_portven=bool(m.get("portven_shape")), portven_shape=m.get("portven_shape", ""), portven_spacing=m.get("portven_spacing", ""),
                     in_pv_cohort=pid in use, final_status=u.get("final_status", ""), usable_final=u.get("usable_final", ""),
                     main_analysis_usable=u.get("main_analysis_usable", ""), pending_visual_16=pid in pend16, box_npz=box, box_source=src))
out = os.path.join(M, "case_list_all.csv")
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(out, "rows", len(rows), "with_pv", sum(r["has_portven"] for r in rows), "with_box", sum(bool(r["box_npz"]) for r in rows), "pending_visual", sum(r["pending_visual_16"] for r in rows))
