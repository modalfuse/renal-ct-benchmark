#!/usr/bin/env python3
"""Crop a PV ROI around the mask bounding box (or the box if the mask is empty).

Writes <out>/roi/<pid>_img.nii.gz (int16 HU), <pid>_mask.nii.gz (uint8),
<pid>_box.nii.gz when a box exists, and roi_manifest.csv.
Existing outputs are skipped unless --overwrite. No network.
"""
import argparse, os, sys, csv, json, time, logging, traceback
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import read_cases, index_inputs, load_pv, load_box, affine

ap = argparse.ArgumentParser()
ap.add_argument("--data-root", required=True); ap.add_argument("--meta-dir", required=True)
ap.add_argument("--seg-dir", required=True, help="seg_pv.py --out-dir (reads masks/)"); ap.add_argument("--out-dir", required=True)
ap.add_argument("--margin-mm", type=float, default=10.0); ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--pids", default=""); ap.add_argument("--overwrite", action="store_true")
a = ap.parse_args()
R = os.path.join(a.out_dir, "roi"); os.makedirs(R, exist_ok=True); os.makedirs(os.path.join(a.out_dir, "logs"), exist_ok=True); os.makedirs(os.path.join(a.out_dir, "roi_qc"), exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    handlers=[logging.FileHandler(os.path.join(a.out_dir, "logs", "crop_roi.log")), logging.StreamHandler()])

def one(args):
    c, path = args; pid = c["pid"]; qf = os.path.join(a.out_dir, "roi_qc", f"{pid}.json")
    mf0 = os.path.join(a.seg_dir, "masks", f"{pid}.nii.gz")
    stale = os.path.exists(qf) and os.path.exists(mf0) and os.path.getmtime(mf0) > os.path.getmtime(qf)
    if not a.overwrite and not stale and os.path.exists(qf) and json.load(open(qf)).get("status") == "ok": return json.load(open(qf))
    import nibabel as nib
    q = dict(pid=pid, label_5class=c["label_5class"], pending_visual_16=c["pending_visual_16"], final_status=c["final_status"], box_source=c["box_source"])
    try:
        vol, sp = load_pv(path)
        mf = os.path.join(a.seg_dir, "masks", f"{pid}.nii.gz")
        mask = np.asanyarray(nib.load(mf).dataobj).astype(bool) if os.path.exists(mf) else np.zeros(vol.shape, bool)
        box = load_box(os.path.join(a.meta_dir, c["box_npz"]), vol.shape)[0] if c["box_npz"] else None
        ref, src = (mask, "mask") if mask.any() else ((box, "box") if box is not None and box.any() else (None, ""))
        if ref is None: raise ValueError("empty mask and no box")
        nz = np.array(np.nonzero(ref)); m = np.ceil(a.margin_mm / np.array(sp)).astype(int)
        lo = np.maximum(nz.min(1) - m, 0); hi = np.minimum(nz.max(1) + m + 1, vol.shape)
        s = tuple(slice(l, h) for l, h in zip(lo, hi))
        A = affine(sp).copy(); A[:3, 3] = A[:3, :3] @ lo
        nib.save(nib.Nifti1Image(vol[s].astype(np.int16), A), os.path.join(R, f"{pid}_img.nii.gz"))
        nib.save(nib.Nifti1Image(mask[s].astype(np.uint8), A), os.path.join(R, f"{pid}_mask.nii.gz"))
        if box is not None: nib.save(nib.Nifti1Image(box[s].astype(np.uint8), A), os.path.join(R, f"{pid}_box.nii.gz"))
        q.update(status="ok", roi_from=src, lo_xyz="x".join(map(str, lo)), hi_xyz="x".join(map(str, hi)), roi_shape="x".join(map(str, hi - lo)),
                 spacing="x".join(f"{v:.4g}" for v in sp), mask_vox=int(mask.sum()), img=f"roi/{pid}_img.nii.gz", mask=f"roi/{pid}_mask.nii.gz")
    except Exception as e:
        q.update(status="error", error=f"{type(e).__name__}: {e}"); logging.info(traceback.format_exc())
    json.dump(q, open(qf, "w"), ensure_ascii=False); logging.info(f"{pid} {q['status']} {q.get('roi_shape','')} {q.get('error','')}")
    return q

if __name__ == "__main__":
    cases = [c for c in read_cases(a.meta_dir) if c["has_portven"] == "True"]
    if a.pids:
        want = set(open(a.pids).read().split()) if os.path.exists(a.pids) else set(a.pids.split(","))
        cases = [c for c in cases if c["pid"] in want]
    idx = index_inputs(a.data_root)
    jobs = [(c, idx[c["pid"]]) for c in cases if c["pid"] in idx]
    logging.info(f"{len(jobs)} cases to crop ({len(cases)-len(jobs)} missing inputs)")
    with Pool(a.workers) as p: p.map(one, jobs, chunksize=1)
    QD = os.path.join(a.out_dir, "roi_qc")
    rows = [json.load(open(os.path.join(QD, f))) for f in sorted(os.listdir(QD)) if f.endswith(".json")]
    keys = []; [keys.append(k) for r in rows for k in r if k not in keys]
    with open(os.path.join(a.out_dir, "roi_manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    logging.info(f"roi_manifest.csv rows={len(rows)} ok={sum(r['status']=='ok' for r in rows)}")
