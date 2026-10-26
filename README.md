# Renal CT Benchmark: public five-class renal-tumor CT subtyping

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23668646.svg)](https://doi.org/10.5281/zenodo.23668646)

Repository: [https://github.com/modalfuse/renal-ct-benchmark](https://github.com/modalfuse/renal-ct-benchmark)  
Release: [v1.0.0](https://github.com/modalfuse/renal-ct-benchmark/releases/tag/v1.0.0)  
Zenodo: [https://doi.org/10.5281/zenodo.23668646](https://doi.org/10.5281/zenodo.23668646)

Renal CT Benchmark asks whether fully public CT is sufficient to train a reproducible cross-institutional five-class renal-tumor classifier, and where the performance ceiling lies. Training uses UCSF-RMaC portal-venous cases. External testing uses RCC-AID (three classes) and KiTS23 (five classes). The primary metric is balanced accuracy.

This package contains the frozen five-fold file, the L0–L4 scripts, and the shell wrappers that call them. Images, masks, ROI volumes, large feature tables, and foundation-model weights are not distributed.

## Repository contents

- `data/ucsf_pv_stratified_split_5fold.csv`: 532 rows; columns `pid`, `label_5class`, and `fold` (0–4). Class counts: ccRCC 347, oncocytoma 76, pRCC 60, chRCC 30, AML 19.
- `src/`: L0 size logistic regression, L2 radiomics, L3 ResNet training and external evaluation, L4 linear probes, plus the shared loaders, segmentation, and ROI crop.
- `scripts/`: `run_cv.sh`, `run_ext.sh`, and `run_l4.sh`.
- `CONTENTS.md`: file list.

## Frozen protocol

The recorded seed is `20261024`. That seed applies to augmentation and weight initialization. Fold membership is the CSV above. Do not rebuild it with `StratifiedKFold`.

Also report accuracy, macro-F1 with `zero_division=0`, and one-versus-rest AUC. Cross-validation statistics are the mean and the population standard deviation (`ddof=0`) across the five folds. External predictions average the five fold-wise probability vectors, then take argmax. For RCC-AID, keep the first three components of that averaged vector (ccRCC, pRCC, chRCC) and renormalize each row with a denominator floor of `1e-8`. KiTS23 keeps the five-class vector. Uncertainty for an external difference uses a class-stratified bootstrap of 4,000 resamples with seed `20261026`.

Image and weight paths come from the environment or from command-line arguments. L4 readers require `L4_ROOT`, `L4_SPLIT`, `L4_QC`, `L4_EXT`, and `L4_ROI`.

- **L0.** Multinomial logistic regression on `log(mask volume)` and `log(axis-aligned longest diameter)`, `class_weight="balanced"`, `C=1.0`.
- **L2.** PyRadiomics original families only (shape, firstorder, glcm, glrlm, glszm, ngtdm, gldm), `binWidth=25` HU, no resampling and no intensity normalization. XGBoost uses `n_estimators=300`, `max_depth=4`, `learning_rate=0.1`, `subsample=0.8`, `colsample_bytree=0.8`, and balanced sample weights.
- **L3.** MONAI 3D ResNet on 96³ dual-channel inputs: HU clipped to [−200, 300] and scaled to [0, 1], plus the tumor mask. AdamW (`lr=1e-4`, `weight_decay=1e-4`), `CosineAnnealingLR` with `T_max=100`, batch size 8, 100 epochs. Inverse-frequency class weights are computed inside each training fold only, `w_c = n / (5 n_c)`. The reported runs are ResNet18 with weighted cross-entropy, ResNet50 with weighted cross-entropy (R1), ResNet50 with focal loss (`gamma=2`, `alpha_c=w_c`; R2), and ResNet50 with unweighted cross-entropy plus `WeightedRandomSampler` (R3). Select each fold checkpoint by validation balanced accuracy.
- **L4.** Frozen embeddings and a logistic-regression head only (`C=1.0`, `class_weight="balanced"`). Fit median imputation and `StandardScaler` on the training fold. Do not include volume or diameter. Do not fine-tune. The three encoders are MMM/UMedPT (768-d), FMCIB (4096-d), and CT-FM (512-d).

```bash
# L3 cross-validation. Set SEG_OUT, L4_SPLIT, LABELS, and OUT.
bash scripts/run_cv.sh

# External test. Set CKPT_DIR, RCC_ROOT, and KITS_ROOT.
bash scripts/run_ext.sh

# One L4 encoder: mmm, ctfm, or fmcib. Set L4_ROOT and the table paths above.
bash scripts/run_l4.sh mmm
```

The analysis denominators used with the frozen fold file are 532 UCSF-RMaC portal-venous cases, 91 RCC-AID patients, and 427 KiTS23 cases. Build those cohorts from the public downloads.

## Data and weights

This repository does not contain images, masks, clinical tables, or foundation-model weights. Obtain each source from its provider. Access terms stay with that provider.

| Cohort or weight | Licence | Where to obtain it |
| --- | --- | --- |
| UCSF-RMaC | CC BY 4.0 | doi:10.58078/C2WC74 |
| RCC-AID | Annotations CC BY 4.0; CT images keep the TCIA terms of TCGA-KIRC, TCGA-KIRP, and TCGA-KICH | MELBA doi:10.59275/j.melba.2026-2dg2; Zenodo 19630298 and 20719257 |
| KiTS23 | CC BY-NC-SA 4.0 (non-commercial) | doi:10.5281/zenodo.7840133 |
| MMM / UMedPT checkpoint | Fraunhofer non-commercial research licence; do not redistribute the file | Code `FraunhoferMEVIS/MedicalMultitaskModeling`; Zenodo record 11383543 |
| FMCIB weights | CC BY-NC; do not redistribute the file | Code `AIM-Harvard/foundation-cancer-image-biomarker` (MIT); doi:10.5281/zenodo.10528450 |
| CT-FM feature extractor | Official model licence | Code `project-lighter/CT-FM`; Hugging Face `project-lighter/ct_fm_feature_extractor` |

## Citation

Cite this software archive as:

> Liu Q. Renal CT Benchmark: public five-class renal-tumor CT subtyping. Zenodo. https://doi.org/10.5281/zenodo.23668646

Citation metadata are also provided in `CITATION.cff`.

## Licensing

- Source code and this documentation: MIT License (`LICENSE`).
- The frozen fold file lists public case identifiers and fold assignments produced for this benchmark. It is distributed with the code under the same MIT License.
- Images and third-party weights are not included. Their licences remain those of UCSF-RMaC (CC BY 4.0), RCC-AID (annotations CC BY 4.0), KiTS23 (CC BY-NC-SA 4.0), and the foundation-model providers named above.
