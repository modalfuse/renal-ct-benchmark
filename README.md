# Public five-class renal-tumor CT benchmark

This repository asks whether fully public CT is sufficient to train a reproducible cross-institutional five-class renal-tumor classifier, and where the performance ceiling lies. Training uses UCSF-RMaC portal-venous cases. External testing uses RCC-AID and KiTS23.

Version 1.0.0: https://github.com/modalfuse/renal-ct-benchmark. Archive: https://doi.org/10.5281/zenodo.23668646. The code is MIT. Cite the accompanying manuscript for the scientific findings.

## Data to obtain separately

Download each cohort from its public source. This tree does not contain images, masks, or clinical tables.

| Cohort | Public name | Licence | Where to obtain it |
| --- | --- | --- | --- |
| Training | UCSF-RMaC (UCSF Renal Mass CT Dataset) | CC BY 4.0 | doi:10.58078/C2WC74 (UCSF Ci2) |
| External test | RCC-AID | Annotations CC BY 4.0; CT images keep the TCIA terms of TCGA-KIRC, TCGA-KIRP, and TCGA-KICH | MELBA doi:10.59275/j.melba.2026-2dg2; Zenodo records 19630298 and 20719257 |
| External test | KiTS23 (2023 Kidney and Kidney Tumor Segmentation Challenge) | CC BY-NC-SA 4.0 (non-commercial) | doi:10.5281/zenodo.7840133; code repository `neheller/kits23` |

The analysis denominators used with the frozen fold file are 532 UCSF-RMaC portal-venous cases, 91 RCC-AID patients evaluated as three classes, and 427 KiTS23 cases evaluated as five classes. Build those cohorts from the public downloads. This tree ships only the fold file below.

## Frozen folds

`data/ucsf_pv_stratified_split_5fold.csv` has 532 rows and the columns `pid`, `label_5class`, and `fold` (fold ids 0–4). The recorded seed is `20261024`. That seed applies to augmentation and weight initialization. Fold membership is this file. Do not rebuild it with `StratifiedKFold`.

Class counts in the file: ccRCC 347, oncocytoma 76, pRCC 60, chRCC 30, AML 19.

## Metrics

The primary metric is balanced accuracy. Also report accuracy, macro-F1 with `zero_division=0`, and one-versus-rest AUC.

Cross-validation statistics are the mean and the population standard deviation (`ddof=0`) across the five folds. External predictions average the five fold-wise probability vectors, then take argmax. For RCC-AID, keep the first three components of that averaged vector (ccRCC, pRCC, chRCC) and renormalize each row with a denominator floor of `1e-8`. KiTS23 keeps the five-class vector. Uncertainty for an external difference uses a class-stratified bootstrap of 4,000 resamples with seed `20261026`.

## Reproducing L0–L4

Scripts are in `src/`. Shell wrappers are in `scripts/`. They follow the frozen settings below. Image and weight paths come from the environment or from command-line arguments. The fold file shipped here is `data/ucsf_pv_stratified_split_5fold.csv`.

- **L0.** Multinomial logistic regression on `log(mask volume)` and `log(axis-aligned longest diameter)`, `class_weight="balanced"`, `C=1.0`.
- **L2.** PyRadiomics original families only (shape, firstorder, glcm, glrlm, glszm, ngtdm, gldm), `binWidth=25` HU, no resampling and no intensity normalization. XGBoost uses `n_estimators=300`, `max_depth=4`, `learning_rate=0.1`, `subsample=0.8`, `colsample_bytree=0.8`, and balanced sample weights.
- **L3.** MONAI 3D ResNet on 96³ dual-channel inputs: HU clipped to [−200, 300] and scaled to [0, 1], plus the tumor mask. AdamW (`lr=1e-4`, `weight_decay=1e-4`), `CosineAnnealingLR` with `T_max=100`, batch size 8, 100 epochs. Inverse-frequency class weights are computed inside each training fold only, `w_c = n / (5 n_c)`. The reported runs are ResNet18 with weighted cross-entropy, ResNet50 with weighted cross-entropy (R1), ResNet50 with focal loss (`gamma=2`, `alpha_c=w_c`; R2), and ResNet50 with unweighted cross-entropy plus `WeightedRandomSampler` (R3). Select each fold checkpoint by validation balanced accuracy.
- **L4.** Frozen embeddings and a logistic-regression head only (`C=1.0`, `class_weight="balanced"`). Fit median imputation and `StandardScaler` on the training fold. Do not include volume or diameter. Do not fine-tune. The three encoders are MMM/UMedPT (768-d), FMCIB (4096-d), and CT-FM (512-d).

## Weights: download, do not commit

This tree does not contain foundation-model weights.

- **MMM / UMedPT.** Code: `https://github.com/FraunhoferMEVIS/MedicalMultitaskModeling`. Checkpoint: Zenodo record 11383543. The Fraunhofer non-commercial research licence allows research evaluation and does not allow redistributing copies. Do not place the checkpoint in this repository.
- **FMCIB.** Code: `https://github.com/AIM-Harvard/foundation-cancer-image-biomarker` (MIT). Weights: the Zenodo record named by that repository, doi:10.5281/zenodo.10528450 (`model_weights.torch`). The weights are CC BY-NC: research use is allowed, and this repository does not redistribute the weight file.
- **CT-FM.** Code: `https://github.com/project-lighter/CT-FM`. Feature-extractor weights: the official model page `project-lighter/ct_fm_feature_extractor` on Hugging Face. Do not commit those weights here.

## What this tree contains

- This `README.md` and `CONTENTS.md`
- `data/ucsf_pv_stratified_split_5fold.csv`
- `src/` for L0–L4, segmentation, training, and external evaluation
- `scripts/run_cv.sh`, `scripts/run_ext.sh`, and `scripts/run_l4.sh`

## What this tree does not contain

- Foundation-model weights, including MMM and FMCIB
- Raw CT, masks, or ROI volumes
- Large feature tables
- Any internal cohort that is outside the three public datasets named above
- One-off install wrappers that hard-coded a server directory
