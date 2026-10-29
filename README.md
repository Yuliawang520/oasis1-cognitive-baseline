# OASIS-1 cognitive-score baseline: code and data

Code, the subject-level cross-validation split, and out-of-fold model predictions for

> Y. Cao, J. Fu, and S. Ling, "An Input-Attribution Framework for Dementia Classifiers: Cognitive Score and Age, Not Architecture, Explain MRI Model Accuracy," manuscript, 2026.

The paper attributes the accuracy of thirteen dementia classifiers to four sources: the cognitive score given as input, the age of the subjects, the images and the model architecture. On 226 OASIS-1 subjects (133 non-demented, 93 demented) and one shared subject-level 5-fold split:

- The MMSE score, with no model fitted, reaches AUC 0.890. After Holm correction it significantly outperforms all six imaging-only models (five ImageNet-pretrained deep networks and a two-volume logistic regression) and both ANFIS configurations, and no model significantly outperforms it.
- Adding hippocampal and whole-brain volume to the MMSE is statistically detectable (likelihood-ratio p = 0.0005), but the gain in discrimination is small and uncertain: AUC +0.025 (95% CI −0.003 to 0.053), and +0.035 (0.007 to 0.067) in subjects aged 60 or over.
- A logistic regression on the ANFIS's three features (4 parameters) significantly outperforms the 126-parameter ANFIS (0.907 vs. 0.823, p = 0.0005).
- Age alone, with no model fitted, reaches AUC 0.665 (0.536 in subjects aged 60 or over). After Holm correction (family F4), only one of the five deep networks significantly exceeds it on the full cohort. Within the non-demented subjects, the network outputs correlate with age (Spearman ρ 0.55–0.82), whereas the age-residualised LR-2 does not (−0.17).
- Restricted to subjects aged 60 or over, the deep networks fall from AUC 0.68–0.76 to 0.61–0.68, consistent with an age confound.
- A linear probe of frozen BrainIAC features (a 3D foundation model for brain MRI; nothing trained on OASIS-1) reaches AUC 0.744 (0.669 in subjects aged 60 or over), within the range of the five deep networks: 0.146 below the MMSE score, 0.079 above age alone (Holm p = 0.051) and level with the two-volume LR-2. Its output correlates with age within the non-demented subjects (ρ = 0.64). Added to the MMSE by nested stacking, it raises the AUC by 0.031 (95% CI 0.007 to 0.059), to 0.914.
- Ablations of the ANFIS: rule bases of 8, 27 and 64 rules give AUCs of 0.814, 0.823 and 0.806, so the rule count matters little. A binary target in place of the CDR regression raises the AUC to 0.857, or 0.874 without SMOTE. The best variants still trail the logistic regression (p ≤ 0.018).
- Across 100 repeated five-fold splits, every comparison among the tabular models keeps its direction in at least 98% of the splits.
- Decision curves: adding the volumes to the MMSE raises the net benefit by at most 0.04 at thresholds from 0.1 to 0.6.

## Contents

| Path | What it is |
| --- | --- |
| `notebooks/` | Google Colab notebooks that produced every model in the paper (see [Reproducing the paper](#reproducing-the-paper)) |
| `scripts/evaluate.py` | Refits the four fitted baselines, computes the two zero-parameter references (MMSE alone, age alone) and reproduces the paper's main tables, Fig. 1 and Fig. S1, checking each value against `scripts/paper_values.json` |
| `scripts/robustness.py` | ANFIS ablations, 100 repeated five-fold splits and decision curves (Tables S2, S3 and S6 and Fig. S2 of the paper's Supplementary Material) |
| `scripts/anfis.py` | The ANFIS of the training notebook, with switches for the ablations; reproduces the released ANFIS outputs |
| `scripts/brainiac_features.py` | Registers the FreeSurfer brain masks to the BrainIAC template and extracts frozen BrainIAC (ViT-B) features; run in Colab by `notebooks/brainiac_probe_features_colab.ipynb` |
| `scripts/probe.py` | Linear probe of the frozen BrainIAC features on the shared split (3D imaging-only baseline): out-of-fold scores, family F5, nested MMSE + probe fusion and age diagnostics |
| `scripts/adni_build_volumes.py` | Builds the ADNI baseline volume table (hippocampus, BrainSegVol, ICV) from the ADSP PHC FreeSurfer table of the ADNIMERGE2 R data package |
| `scripts/adni_replication.py` | External replication of the tabular attribution on the ADNI1 baseline cohort (zero-parameter references, LR-1/2/3, CART-3, ANFIS; families A1–A4). ADNI data cannot be redistributed; the committed files in `results/adni/` hold aggregate statistics only, and the fold assignment with ADNI subject IDs is git-ignored |
| `data/folds.csv` | The subject-level 5-fold assignment shared by all thirteen models |
| `data/oof_predictions.csv` | Out-of-fold outputs of both ANFIS configurations and the five deep baselines |
| `data/oof_probe_*.csv` | Out-of-fold scores of the BrainIAC probe (primary analysis, the original penalty grid and the mean-pooled variant) and of the MMSE + probe fusion |
| `results/` | Output of `evaluate.py` (tables as CSV including `table_f4_age.csv`, `results.json`, `fig_roc.pdf` and `fig_calib.pdf`, which are Fig. 1 and Fig. S1), of `robustness.py` (`table_ablation.csv`, `table_dca.csv`, `table_repeated_cv_*.csv`, `robustness.json` and `fig_dca.pdf`, which is Fig. S2) and of `probe.py` (`probe_*.json`, `table_probe_*.csv`; Table S5), plus ANFIS rule tables and confusion matrices, and deep-baseline confusion matrices and training curves |

### `data/` columns

- `subject_id`: OASIS-1 session ID (first session, `_MR1`).
- `fold`: test fold (0–4). Every prediction for a subject comes from the model trained on the other four folds.
- `anfis_gt_severity`, `anfis_extracted_severity`: raw ANFIS output on the CDR scale, with FreeSurfer or HippoDeep hippocampal volume as input.
- `densenet121`, `vgg16`, `resnet50`, `deit_small`, `swin_t`: probability of dementia, averaged over 24 axial slices per subject.

## Not included

The [OASIS Data Use Agreement](https://sites.wustl.edu/oasisbrains/) does not allow OASIS data to be redistributed in whole or in part. This repository therefore contains no images, no subject-level clinical or demographic variables (age, sex, MMSE, CDR, diagnosis) and no brain volumes. The predictions of LR-1, of the raw MMSE ranking and of the age-only ranking are also withheld, because they would disclose MMSE values or ages; `evaluate.py` regenerates them. Apply for OASIS-1 access and rebuild the inputs with the notebooks. `results/` contains only aggregate statistics.

Trained model weights are not included.

## Reproducing the paper

1. Obtain OASIS-1 from <https://sites.wustl.edu/oasisbrains/>: the FreeSurfer-processed release (discs 1–11, providing `mri/brainmask.mgz` and `stats/aseg.stats` for each session) and the cross-sectional demographics spreadsheet. The notebooks expect them in `MyDrive/OASIS_Freesurfer/` on Google Drive; change `DRIVE_ROOT` and `OASIS_DIR` at the top of each notebook otherwise.
2. Run the notebooks in Colab, in this order:
   1. `alzheimer_feature_extraction_colab.ipynb`: hippocampal volumes from `brainmask.mgz` with [HippoDeep](https://github.com/bthyreau/hippodeep_pytorch).
   2. `alzheimer_labels_merge_colab.ipynb`: merges age, MMSE and CDR, parses `aseg.stats`, and writes `oasis_master_features_labels.csv`.
   3. `alzheimer_anfis_train_colab.ipynb`: trains ANFIS-on-GT and ANFIS-on-extracted. This notebook creates the shared split (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on the 226 subjects with complete features).
   4. `alzheimer_eval_anfis_colab.ipynb`: ANFIS confusion matrices.
   5. `alzheimer_baseline_densenet121_oasis_colab.ipynb` first, because it writes the axial slices that the other deep baselines reuse; then the VGG-16, ResNet-50, DeiT-Small and Swin-T notebooks. All of them read the split from the ANFIS-on-GT output.
3. Reproduce the tables and figures:

   ```bash
   pip install -r requirements.txt
   python scripts/evaluate.py --cohort path/to/oasis_master_features_labels.csv \
       --demographics path/to/oasis_cross-sectional.xlsx
   ```

   The script prints every table and reports how many values match the paper: all 361 should match, or 347 without `--demographics`, which only the cohort table needs. It reads the probe's out-of-fold scores from `data/`, so step 5 is not needed to reproduce the main tables. Tested with Python 3.12, numpy 2.2, pandas 2.3, scipy 1.15, scikit-learn 1.8 and matplotlib 3.10.
4. Reproduce the robustness analyses (needs torch and imbalanced-learn; about 25 minutes on a laptop CPU, because the repeated splits retrain the ANFIS 1000 times):

   ```bash
   python scripts/robustness.py --cohort path/to/oasis_master_features_labels.csv
   ```

   The script first re-trains the paper's ANFIS-on-GT and prints how closely it matches the released outputs. Tested with torch 2.8 (CPU) and imbalanced-learn 0.14.
5. BrainIAC linear probe (3D imaging-only baseline). Download `BrainIAC.ckpt` from the [BrainIAC repository](https://github.com/AIM-KannLab/BrainIAC) (non-commercial academic licence), run `notebooks/brainiac_probe_features_colab.ipynb` to register the 226 brain masks to the BrainIAC template and extract the frozen features (`brainiac_features.csv`, BrainIAC's own first-token representation; `brainiac_features_meanpool.csv`, mean of the 216 patch tokens), then fit the probe locally:

   ```bash
   python scripts/probe.py --cohort path/to/oasis_master_features_labels.csv --features path/to/brainiac_features.csv
   ```

   The feature CSVs are derived from OASIS images and are not part of the repository; the out-of-fold probe scores in `data/` are.
6. ADNI replication (needs an approved IDA account). Download the ADNIMERGE2 R data package from the IDA and extract it; build the clinical metadata of the ADNI1 "Complete 1Yr 1.5T" participants (`subject_id`, `timepoint`, `age_exact`, `mmscore`, `cdglobal`) as in the longitudinal ADNI notebooks; then:

   ```bash
   pip install pyreadr
   python scripts/adni_build_volumes.py --package path/to/ADNIMERGE2 --clinical path/to/clinical_metadata.csv --out adni_volumes_bl.csv
   python scripts/adni_replication.py --clinical path/to/clinical_metadata.csv --adnimerge adni_volumes_bl.csv --anfis
   ```

   The committed outputs in `results/adni/` contain aggregate statistics only (Table VI and Table S7 of the paper); `adni_folds.csv`, which lists ADNI subject IDs, is git-ignored and is reproducible from the fixed random state.

## Analysis notes

- **ANFIS probabilities.** The ANFIS regresses the CDR score. For accuracy, sensitivity, specificity, the Brier score and the calibration curve, its output is mapped to a probability by a logistic (Platt) recalibration fitted, for each test fold, on the out-of-fold outputs of the other four folds. AUC is computed on the raw output.
- **Primary comparisons.** Holm's procedure is applied within four named families: (F1) the MMSE score against each of the ten models with imaging inputs, repeated with the fitted LR-1 as the reference; (F2) LR-3 against LR-1, also with a likelihood-ratio test of the two imaging coefficients; (F3) LR-3 against ANFIS-on-GT; (F4) age alone against each of the five deep networks, on all subjects and on those aged 60 or over, adjusted within each cohort. Other p-values are exploratory and unadjusted.
- **Age alone.** The thirteenth model ranks subjects by age in years with no fitting. As a diagnostic of the age attribution, the Spearman correlation between each model output and age is computed within the 133 non-demented subjects, whose constant label cannot drive the correlation.
- **Pooling.** AUC is computed on the pooled out-of-fold predictions; per-fold mean ± SD is reported alongside, because pooling penalises fitted models relative to the unfitted MMSE ranking.
- **Age sensitivity.** All comparisons are repeated on the 189 subjects aged 60 or over, using the same out-of-fold predictions.
- **Baselines.** All baselines use train-fold-only age residualisation of the volumes and train-fold-only standardisation, as the ANFIS does. CART-3 uses `random_state=0`.
- **Swin-T.** The Swin-T notebook up-weights the minority class by a further factor of 1.5 and early-stops on balanced accuracy, because with plain inverse-frequency weighting it collapsed to predicting the majority class.
- **Ablations.** `scripts/anfis.py` reproduces the released ANFIS-on-GT outputs (AUC 0.8232; largest per-subject difference 9.6e-4, from GPU versus CPU arithmetic). Each variant changes the stated settings only and keeps the original optimiser settings. Variants with SMOTE are repeated with ten SMOTE random states. Statistics use the row order of `evaluate.py`, so the original configuration reproduces comparison F3 exactly.
- **Repeated splits.** 100 stratified five-fold splits (`random_state` 0–99) for the five interpretable baselines, ANFIS-on-GT and the minimal ANFIS variant. The deep networks are not retrained.
- **BrainIAC probe.** Preprocessing follows BrainIAC's released pipeline (N4, rigid registration to its `temp_head.nii.gz` template with Mattes mutual information, whole volume resized to 96³, z-scored over non-zero voxels); HD-BET is skipped because `brainmask.mgz` is already skull-stripped. The probe is an L2 logistic regression on the standardised 768-dimensional features, fitted on the four training folds of the shared split with the penalty chosen by an inner five-fold cross-validation. Pre-specified comparisons (family F5, Holm within three per cohort): probe vs MMSE alone, vs age alone and vs LR-2, on all subjects and on those aged 60 or over. The MMSE + probe fusion uses nested stacking, so no training subject receives a probe score from a model that saw it. BrainIAC's pretraining pool includes ADNI, OASIS-2, OASIS-3 and MIRIAD, but not OASIS-1; the pretraining used no labels. The penalty grid specified before the features existed (1e-4 to 10) was chosen at its lower boundary in every fold, so the primary analysis uses the grid 1e-6 to 10 and the original grid is kept as a sensitivity analysis (`--c-grid 1e-4,1e-3,1e-2,1e-1,1,10`); the released checkpoint is a MONAI ViT-B without a class token, so BrainIAC's representation is the first token of the final layer, and the mean of the 216 patch tokens is a second sensitivity variant.
- **Decision curves.** Net benefit of LR-1, LR-3, LR-2 and the recalibrated ANFIS-on-GT on the shared split, with paired bootstrap CIs for LR-3 minus LR-1 at thresholds of 0.1 to 0.6. Net benefit is computed at the cohort prevalence (41%), which reflects OASIS-1 recruitment rather than a clinical population.
- Comments in the notebooks are in Chinese.

## Acknowledgment

Data were provided by OASIS. OASIS-1: Cross-Sectional: Principal Investigators: D. Marcus, R. Buckner, J. Csernansky, J. Morris; P50 AG05681, P01 AG03991, P01 AG026276, R01 AG021910, P20 MH071616, U24 RR021382.

OASIS-1 reference: D. S. Marcus et al., "Open Access Series of Imaging Studies (OASIS): Cross-sectional MRI data in young, middle aged, nondemented, and demented older adults," *J. Cogn. Neurosci.*, vol. 19, no. 9, pp. 1498–1507, 2007. <https://doi.org/10.1162/jocn.2007.19.9.1498>

## Citation

```bibtex
@misc{cao2026mri,
  author = {Cao, Yu and Fu, Jinrong and Ling, Steve},
  title  = {An Input-Attribution Framework for Dementia Classifiers: Cognitive Score and Age, Not Architecture, Explain {MRI} Model Accuracy},
  note   = {Manuscript},
  year   = {2026}
}
```

## License

The code and the files in `data/` and `results/` are released under the MIT License (see `LICENSE`). OASIS data are not part of this repository and remain subject to the OASIS Data Use Agreement.
