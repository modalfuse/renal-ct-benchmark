# Contents

This file lists what is in version 1.0.0.

## Included now

| Path | What it is |
| --- | --- |
| `README.md` | Study question, public data sources, frozen fold, metrics, L0–L4 settings, and the weight-download policy |
| `CONTENTS.md` | This list |
| `data/ucsf_pv_stratified_split_5fold.csv` | 532 rows; columns `pid`, `label_5class`, `fold` only |
| `src/l0_size_baseline.py` | L0 size logistic regression |
| `src/l2_ucsf_extract.py` | PyRadiomics feature extraction |
| `src/l2_radiomics.py` | L2 frozen XGBoost |
| `src/l3_backbone.py` | L3 ResNet50 runs R1, R2, and R3 |
| `src/train_cls.py` | ResNet18 fixed-fold training |
| `src/eval_ext.py` | RCC-AID and KiTS23 external evaluation |
| `src/common.py` | Shared PV load and box decode |
| `src/build_case_list.py` | Build the case table from public metadata |
| `src/seg_pv.py` | Portal-venous segmentation |
| `src/crop_roi.py` | ROI crop |
| `src/l4_tables.py` | Read the fold, QC, and external-task tables from the environment |
| `src/l4_store.py` | Append feature rows and error rows |
| `src/l4_probe.py` | Linear probe |
| `src/l4_mmm_run.py` | MMM embedding and probe |
| `src/l4_ctfm_run.py` | CT-FM embedding and probe |
| `src/l4_fmcib_run.py` | FMCIB embedding and probe |
| `src/l4_fmcib_singleton.py` | FMCIB single-case check |
| `scripts/run_cv.sh` | Train with `SEG_OUT`, `L4_SPLIT`, `LABELS`, and `OUT` |
| `scripts/run_ext.sh` | External eval with `CKPT_DIR`, `RCC_ROOT`, and `KITS_ROOT` |
| `scripts/run_l4.sh` | One of `mmm`, `ctfm`, or `fmcib` under `L4_ROOT` |

Paths are arguments or environment variables. The tree does not set a machine directory.

L4 readers require `L4_ROOT`, `L4_SPLIT`, `L4_QC`, `L4_EXT`, and `L4_ROI`.

## Not in this tree

- Foundation-model weight files. README gives the official download locations.
- Raw images, masks, and ROI archives.
- Feature matrices and prediction tables from the experiment runs.
- One-off install wrappers that named a server directory.
