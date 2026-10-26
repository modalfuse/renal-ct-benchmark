"""Shared helpers for seg_pv.py and crop_roi.py: case table, PV load, and box decode. No network."""
import csv, os, glob, json
import numpy as np

def read_cases(meta_dir):
    with open(os.path.join(meta_dir, "case_list_all.csv"), encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def index_inputs(data_root):
    """Recursively find <pid>.hdf5/.h5/.hdf or <pid>.npz (keys portven and portven_pixdim) under data_root."""
    idx = {}
    for ext in ("hdf5", "h5", "hdf", "npz"):
        for p in glob.glob(os.path.join(data_root, "**", f"*.{ext}"), recursive=True):
            stem = os.path.basename(p).rsplit(".", 1)[0]
            if ext == "npz" and "_" in stem:
                continue
            idx.setdefault(stem, p)
    return idx

def load_pv(path):
    """Return (vol int16 [X,Y,Z], spacing (x,y,z) mm). Out-of-FOV voxels stay at -3024."""
    if path.endswith(".npz"):
        z = np.load(path)
        return z["portven"].astype(np.int16), tuple(float(v) for v in z["portven_pixdim"])
    import h5py
    with h5py.File(path, "r") as h:
        v = h["portven"][()]
        sp = tuple(float(x) for x in h.attrs["portven_pixdim"])
    return np.clip(np.rint(v), -32768, 32767).astype(np.int16), sp

def load_box(npz_path, shape=None):
    """Decode a packed box npz onto a boolean mask on the PV grid.

    box_bits = np.packbits(crop.ravel()). crop has shape box_crop_shape.
    bbox_lo is the corner in (x, y, z) voxels.
    """
    z = np.load(npz_path, allow_pickle=True)
    cs = tuple(int(v) for v in z["box_crop_shape"]); lo = [int(v) for v in z["bbox_lo"]]
    pv_shape = tuple(int(v) for v in z["pv_shape"])
    if shape is not None and tuple(shape) != pv_shape:
        raise ValueError(f"box grid {pv_shape} != PV {tuple(shape)}")
    crop = np.unpackbits(z["box_bits"], count=int(np.prod(cs))).reshape(cs).astype(bool)
    M = np.zeros(pv_shape, bool)
    M[lo[0]:lo[0]+cs[0], lo[1]:lo[1]+cs[1], lo[2]:lo[2]+cs[2]] = crop
    return M, str(z["method"]) if "method" in z.files else ""

def affine(sp):
    return np.diag([-sp[0], -sp[1], sp[2], 1.0])

def shard(items, i, n):
    return items[i::n]
