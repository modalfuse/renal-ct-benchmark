#!/usr/bin/env python3
"""Offline portal-venous segmentation for UCSF-RMaC.

1. Read the PV volume (hdf5 key portven).
2. Run KiTS23 nnU-Net v2 (3d_lowres_plain, fold_all/checkpoint_best.pth, no mirroring).
   Labels: 1 kidney, 2 tumor, 3 cyst.
3. Read the PV box from case_list_all.csv (reviewed table, then rescue, then original).
   Drop predictions outside the box. Keep tumor label 2 inside the box (2+3 with --include-cyst).
4. If the in-box tumor fraction is below --medsam-thr, run MedSAM (vit_b) per slice and
   take the union with the in-box nnU-Net tumor.
5. Write masks/<pid>.nii.gz (uint8, 1=tumor), nnunet/<pid>_seg.nii.gz, qc/<pid>.json,
   then qc_seg.csv.

Cases without a box keep the whole-image tumor and are flagged no_box.
One process per GPU: --shard i --nshards N. A qc json with status=ok is skipped
unless --overwrite.
"""
import argparse, os, sys, json, time, tempfile, logging, csv, traceback
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import read_cases, index_inputs, load_pv, load_box, affine, shard

ap = argparse.ArgumentParser()
ap.add_argument("--data-root", required=True); ap.add_argument("--meta-dir", required=True); ap.add_argument("--out-dir", required=True)
ap.add_argument("--nnunet-model", required=True, help="…/nnUNetTrainer__nnUNetPlans__3d_lowres_plain_all")
ap.add_argument("--medsam-ckpt", default="", help="medsam_vit_b.pth; empty skips MedSAM")
ap.add_argument("--device", default="cuda", help="cuda / cpu / mps")
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--pids", default="", help="comma-separated pids, or a file with one pid per line; default: every PV case")
ap.add_argument("--box-only", action="store_true", help="only cases that have a box")
ap.add_argument("--medsam-thr", type=float, default=0.2, help="run MedSAM when in-box tumor voxels / box voxels is below this")
ap.add_argument("--include-cyst", action="store_true")
ap.add_argument("--tile-step", type=float, default=0.5)
ap.add_argument("--overwrite", action="store_true"); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--merge-only", action="store_true", help="only merge qc/*.json into qc_seg.csv")
a = ap.parse_args()

O = a.out_dir
for d in ("masks", "nnunet", "qc", "logs", "tmp"): os.makedirs(os.path.join(O, d), exist_ok=True)

def merge_qc():
    rows = []
    for f in sorted(os.listdir(os.path.join(O, "qc"))):
        if f.endswith(".json"):
            rows.append(json.load(open(os.path.join(O, "qc", f))))
    if not rows: return
    keys = []; [keys.append(k) for r in rows for k in r if k not in keys]
    with open(os.path.join(O, "qc_seg.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print("qc_seg.csv", len(rows))
if a.merge_only: merge_qc(); sys.exit(0)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [shard%(s)s] %(message)s".replace("%(s)s", str(a.shard)),
                    handlers=[logging.FileHandler(os.path.join(O, "logs", f"seg_pv_shard{a.shard}.log")), logging.StreamHandler()])
log = logging.info
for k in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
    os.environ.setdefault(k, os.path.join(O, "tmp", k))

import nibabel as nib, torch
if a.device == "mps":  # MPS lacks ConvTranspose3d; optional patch module from MPS_PATCH_DIR
    sys.path.insert(0, os.environ.get("MPS_PATCH_DIR", "")); import mps_conv_transpose; mps_conv_transpose.install()
from scipy import ndimage

cases = read_cases(a.meta_dir)
cases = [c for c in cases if c["has_portven"] == "True"]
if a.box_only: cases = [c for c in cases if c["box_npz"]]
if a.pids:
    want = set(open(a.pids).read().split()) if os.path.exists(a.pids) else set(a.pids.split(","))
    cases = [c for c in cases if c["pid"] in want]
cases = shard(sorted(cases, key=lambda c: c["pid"]), a.shard, a.nshards)
if a.limit: cases = cases[:a.limit]
idx = index_inputs(a.data_root)
log(f"shard cases={len(cases)}; inputs={len(idx)}; device={a.device}")

dev = torch.device(a.device)
predictor = None
def nnunet_predict(vol, sp):
    global predictor
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    if predictor is None:
        predictor = nnUNetPredictor(tile_step_size=a.tile_step, use_gaussian=True, use_mirroring=False,
                                    perform_everything_on_device=a.device != "cpu", device=dev,
                                    verbose=False, verbose_preprocessing=False, allow_tqdm=False)
        predictor.initialize_from_trained_model_folder(a.nnunet_model, use_folds=("all",), checkpoint_name="checkpoint_best.pth")
    with tempfile.TemporaryDirectory(dir=os.path.join(O, "tmp")) as d:
        inp = os.path.join(d, "case_0000.nii.gz")
        nib.save(nib.Nifti1Image(vol, affine(sp)), inp)
        predictor.predict_from_files_sequential([[inp]], [os.path.join(d, "case")], save_probabilities=False, overwrite=True)
        return np.asanyarray(nib.load(os.path.join(d, "case.nii.gz")).dataobj).astype(np.uint8)

medsam = None
def medsam_model():
    global medsam
    if medsam is None:
        from segment_anything import sam_model_registry
        m = sam_model_registry["vit_b"](checkpoint=None)
        sd = torch.load(a.medsam_ckpt, map_location="cpu")
        m.load_state_dict(sd); medsam = m.to(dev).eval()
    return medsam

@torch.no_grad()
def medsam_slices(vol, box):
    """Per slice, each connected rectangle of the box mask is a MedSAM prompt.

    Abdominal window, then per-slice min-max to 0-255, resized to 1024.
    """
    m = medsam_model(); out = np.zeros(box.shape, bool)
    import torch.nn.functional as F
    for z in np.where(box.any((0, 1)))[0]:
        img = np.clip(vol[:, :, z].astype(np.float32), -160, 240)
        img = (img - img.min()) / max(img.max() - img.min(), 1e-6) * 255.0
        H, W = img.shape
        t = torch.from_numpy(np.repeat(img[None], 3, 0)).float()[None].to(dev)
        t = F.interpolate(t, (1024, 1024), mode="bilinear", align_corners=False)
        t = (t - t.min()) / torch.clamp(t.max() - t.min(), min=1e-8)
        emb = m.image_encoder(t)
        lab, n = ndimage.label(box[:, :, z])
        for sl in ndimage.find_objects(lab):
            # Array axes are (X, Y). SAM box is [col0, row0, col1, row1].
            r0, r1, c0, c1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
            bx = torch.tensor([[c0 * 1024 / W, r0 * 1024 / H, c1 * 1024 / W, r1 * 1024 / H]], dtype=torch.float, device=dev)
            se, de = m.prompt_encoder(points=None, boxes=bx[:, None, :], masks=None)
            lo, _ = m.mask_decoder(image_embeddings=emb, image_pe=m.prompt_encoder.get_dense_pe(),
                                   sparse_prompt_embeddings=se, dense_prompt_embeddings=de, multimask_output=False)
            p = torch.sigmoid(F.interpolate(lo, (H, W), mode="bilinear", align_corners=False))[0, 0].cpu().numpy() > 0.5
            out[:, :, z] |= p
    return out & box

for c in cases:
    pid = c["pid"]; qf = os.path.join(O, "qc", f"{pid}.json")
    if not a.overwrite and os.path.exists(qf):
        try:
            if json.load(open(qf)).get("status") == "ok": continue
        except Exception: pass
    t0 = time.time(); q = dict(pid=pid, label_5class=c["label_5class"], pending_visual_16=c["pending_visual_16"], final_status=c["final_status"],
                               box_npz=c["box_npz"], box_source=c["box_source"])
    try:
        if pid not in idx: raise FileNotFoundError("input not found under data root")
        vol, sp = load_pv(idx[pid]); q.update(input=idx[pid], shape="x".join(map(str, vol.shape)), spacing="x".join(f"{v:.4g}" for v in sp))
        segf = os.path.join(O, "nnunet", f"{pid}_seg.nii.gz")
        if os.path.exists(segf) and not a.overwrite:
            seg = np.asanyarray(nib.load(segf).dataobj).astype(np.uint8)
        else:
            seg = nnunet_predict(vol, sp); nib.save(nib.Nifti1Image(seg, affine(sp)), segf)
        if seg.shape != vol.shape: raise ValueError(f"prediction shape {seg.shape} != image {vol.shape}")
        tum_all = np.isin(seg, (2, 3)) if a.include_cyst else (seg == 2)
        q.update(kidney_vox=int((seg == 1).sum()), tumor_vox_all=int((seg == 2).sum()), cyst_vox_all=int((seg == 3).sum()))
        if c["box_npz"]:
            box, method = load_box(os.path.join(a.meta_dir, c["box_npz"]), vol.shape)
            nb = int(box.sum()); inb = tum_all & box
            frac = inb.sum() / max(nb, 1)
            q.update(box_method=method, box_vox=nb, tumor_vox_inbox=int(inb.sum()), tumor_frac_inbox=round(float(frac), 4),
                     kidney_frac_inbox=round(float(((seg == 1) & box).sum() / max(nb, 1)), 4),
                     tumor_vox_outbox=int((tum_all & ~box).sum()))
            final = inb; used = "nnunet"
            if frac < a.medsam_thr:
                if a.medsam_ckpt:
                    ms = medsam_slices(vol, box); final = inb | ms; used = "nnunet+medsam"
                    q.update(medsam_vox=int(ms.sum()), medsam_frac_inbox=round(float(ms.sum() / max(nb, 1)), 4))
                else:
                    used = "nnunet(low_fraction,medsam_off)"
            q.update(mask_method=used, flag="ok" if final.sum() > 0 else "empty_mask")
        else:
            final = tum_all; q.update(mask_method="nnunet_whole", flag="no_box")
        lab, n = ndimage.label(final)
        q.update(mask_vox=int(final.sum()), mask_ml=round(float(final.sum() * np.prod(sp) / 1000), 3), mask_components=int(n))
        nib.save(nib.Nifti1Image(final.astype(np.uint8), affine(sp)), os.path.join(O, "masks", f"{pid}.nii.gz"))
        q["status"] = "ok"
    except Exception as e:
        q.update(status="error", error=f"{type(e).__name__}: {e}"); log(traceback.format_exc())
    q["seconds"] = round(time.time() - t0, 1)
    json.dump(q, open(qf, "w"), ensure_ascii=False)
    log(f"{pid} {q['status']} {q.get('mask_method','')} frac={q.get('tumor_frac_inbox','')} {q['seconds']}s {q.get('error','')}")
if a.nshards == 1: merge_qc()
