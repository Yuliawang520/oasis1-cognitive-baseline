# OASIS-1 cognitive-score baseline: code and data

Code, the subject-level cross-validation split, and out-of-fold model predictions for

> Y. Cao and S. Ling, "How Much Does MRI Add Beyond the MMSE? A Cautionary Benchmark of Deep and Neuro-Fuzzy Dementia Classification on OASIS-1," manuscript, 2026.

On 226 OASIS-1 subjects (133 nondemented, 93 demented) and one shared subject-level 5-fold split:

- The MMSE score, with no model fitted, reaches AUC 0.890. After Holm correction it significantly outperforms all six imaging-only models (five ImageNet-pretrained deep networks and a two-volume logistic regression) and both ANFIS configurations, and no model significantly outperforms it.
- Adding hippocampal and whole-brain volume to the MMSE is statistically detectable (likelihood-ratio p = 0.0005), but the gain in discrimination is small and uncertain: AUC +0.025 (95% CI −0.003 to 0.053), and +0.035 (0.007 to 0.067) in subjects aged 60 or over.
- A logistic regression on the ANFIS's three features (4 parameters) significantly outperforms the 126-parameter ANFIS (0.907 vs. 0.823, p = 0.0005).
- Restricted to subjects aged 60 or over, the deep networks fall from AUC 0.68–0.76 to 0.61–0.68: part of their performance comes from age.

## Contents

| Path | What it is |
| --- | --- |
| `notebooks/` | Google Colab notebooks that produced every model in the paper (see [Reproducing the paper](#reproducing-the-paper)) |
| `scripts/evaluate.py` | Refits the five interpretable baselines and reproduces every table and figure of the paper, checking each value against `scripts/paper_values.json` |
| `data/folds.csv` | The subject-level 5-fold assignment shared by all twelve models |
| `data/oof_predictions.csv` | Out-of-fold outputs of both ANFIS configurations and the five deep baselines |
| `results/` | Output of `evaluate.py` (tables as CSV, `results.json`, Figs. 1 and 2), plus ANFIS rule tables and confusion matrices, and deep-baseline confusion matrices and training curves |

### `data/` columns

- `subject_id`: OASIS-1 session ID (first session, `_MR1`).
- `fold`: test fold (0–4). Every prediction for a subject comes from the model trained on the other four folds.
- `anfis_gt_severity`, `anfis_extracted_severity`: raw ANFIS output on the CDR scale, with FreeSurfer or HippoDeep hippocampal volume as input.
- `densenet121`, `vgg16`, `resnet50`, `deit_small`, `swin_t`: probability of dementia, averaged over 24 axial slices per subject.

## Not included

The [OASIS Data Use Agreement](https://sites.wustl.edu/oasisbrains/) does not allow OASIS data to be redistributed in whole or in part. This repository therefore contains no images, no subject-level clinical or demographic variables (age, sex, MMSE, CDR, diagnosis) and no brain volumes. The predictions of LR-1 and of the raw MMSE ranking are also withheld, because they would disclose MMSE values; `evaluate.py` regenerates them. Apply for OASIS-1 access and rebuild the inputs with the notebooks. `results/` contains only aggregate statistics.

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

   The script prints every table and reports how many values match the paper: all 270 should match, or 256 without `--demographics`, which only the cohort table needs. Tested with Python 3.12, numpy 2.2, pandas 2.3, scipy 1.15, scikit-learn 1.8 and matplotlib 3.10.

## Analysis notes

- **ANFIS probabilities.** The ANFIS regresses the CDR score. For accuracy, sensitivity, specificity, the Brier score and the calibration curve, its output is mapped to a probability by a logistic (Platt) recalibration fitted, for each test fold, on the out-of-fold outputs of the other four folds. AUC is computed on the raw output.
- **Primary comparisons.** Holm's procedure is applied within three named families: (F1) the MMSE score against each of the ten models with imaging inputs, repeated with the fitted LR-1 as the reference; (F2) LR-3 against LR-1, also with a likelihood-ratio test of the two imaging coefficients; (F3) LR-3 against ANFIS-on-GT. Other p-values are exploratory and unadjusted.
- **Pooling.** AUC is computed on the pooled out-of-fold predictions; per-fold mean ± SD is reported alongside, because pooling penalises fitted models relative to the unfitted MMSE ranking.
- **Age sensitivity.** All comparisons are repeated on the 189 subjects aged 60 or over, using the same out-of-fold predictions.
- **Baselines.** All baselines use train-fold-only age residualisation of the volumes and train-fold-only standardisation, as the ANFIS does. CART-3 uses `random_state=0`.
- **Swin-T.** The Swin-T notebook up-weights the minority class by a further factor of 1.5 and early-stops on balanced accuracy, because with plain inverse-frequency weighting it collapsed to predicting the majority class.
- Comments in the notebooks are in Chinese.

## Acknowledgment

Data were provided by OASIS. OASIS-1: Cross-Sectional: Principal Investigators: D. Marcus, R. Buckner, J. Csernansky, J. Morris; P50 AG05681, P01 AG03991, P01 AG026276, R01 AG021910, P20 MH071616, U24 RR021382.

OASIS-1 reference: D. S. Marcus et al., "Open Access Series of Imaging Studies (OASIS): Cross-sectional MRI data in young, middle aged, nondemented, and demented older adults," *J. Cogn. Neurosci.*, vol. 19, no. 9, pp. 1498–1507, 2007. <https://doi.org/10.1162/jocn.2007.19.9.1498>

## Citation

```bibtex
@misc{cao2026mri,
  author = {Cao, Yu and Ling, Steve},
  title  = {How Much Does {MRI} Add Beyond the {MMSE}? A Cautionary Benchmark of Deep and
            Neuro-Fuzzy Dementia Classification on {OASIS-1}},
  note   = {Manuscript},
  year   = {2026}
}
```

## License

The code and the files in `data/` and `results/` are released under the MIT License (see `LICENSE`). OASIS data are not part of this repository and remain subject to the OASIS Data Use Agreement.
