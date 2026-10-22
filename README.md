# OASIS-1 cognitive-score baseline: code and data

Code, the subject-level cross-validation split, and out-of-fold model predictions for

> Y. Cao and S. Ling, "A Single Cognitive Score Outperforms Every MRI-Based Model: A Cautionary Benchmark for Deep and Neuro-Fuzzy Dementia Classification on OASIS-1," manuscript, 2026.

On 226 OASIS-1 subjects (133 nondemented, 93 demented) and one shared subject-level 5-fold split:

- The raw MMSE score, with no model fitted, reaches AUC 0.890 and significantly outperforms every MRI-based model: five ImageNet-pretrained deep networks, two ANFIS configurations, and an imaging-only logistic regression.
- A logistic regression on the ANFIS's three features (4 parameters) reaches AUC 0.907 and significantly outperforms the 126-parameter ANFIS (0.823, DeLong p = 0.0005).
- Hippocampal and whole-brain volume alone reach AUC 0.771.

## Contents

| Path | What it is |
| --- | --- |
| `notebooks/` | Google Colab notebooks that produced every model in the paper (see [Reproducing the paper](#reproducing-the-paper)) |
| `scripts/evaluate.py` | Refits the five interpretable baselines and reproduces Table I, Table II, Fig. 1 and Fig. 2, checking each value against the paper |
| `data/folds.csv` | The subject-level 5-fold assignment shared by all twelve models |
| `data/oof_predictions.csv` | Out-of-fold outputs of both ANFIS configurations and the five deep baselines |
| `results/` | Reproduced Table I and figures, ANFIS rule tables and confusion matrices, deep-baseline confusion matrices and training curves |

### `data/` columns

- `subject_id`: OASIS-1 session ID (first session, `_MR1`).
- `fold`: test fold (0–4). Every prediction for a subject comes from the model trained on the other four folds.
- `anfis_gt_severity`, `anfis_extracted_severity`: raw ANFIS output on the CDR scale, with FreeSurfer or HippoDeep hippocampal volume as input.
- `densenet121`, `vgg16`, `resnet50`, `deit_small`, `swin_t`: probability of dementia, averaged over 24 axial slices per subject.

## Not included

The [OASIS Data Use Agreement](https://sites.wustl.edu/oasisbrains/) does not allow OASIS data to be redistributed in whole or in part. This repository therefore contains no images, no clinical or demographic variables (age, MMSE, CDR, diagnosis) and no brain volumes. Apply for OASIS-1 access and rebuild them with the notebooks.

Trained model weights are not included.

## Reproducing the paper

1. Obtain OASIS-1 from <https://sites.wustl.edu/oasisbrains/>: the FreeSurfer-processed release (discs 1–11, providing `mri/brainmask.mgz` and `stats/aseg.stats` for each session) and the cross-sectional demographics spreadsheet. The notebooks expect them in `MyDrive/OASIS_Freesurfer/` on Google Drive; change `DRIVE_ROOT` and `OASIS_DIR` at the top of each notebook otherwise.
2. Run the notebooks in Colab, in this order:
   1. `alzheimer_feature_extraction_colab.ipynb`: hippocampal volumes from `brainmask.mgz` with [HippoDeep](https://github.com/bthyreau/hippodeep_pytorch).
   2. `alzheimer_labels_merge_colab.ipynb`: merges age, MMSE and CDR, parses `aseg.stats`, and writes `oasis_master_features_labels.csv`.
   3. `alzheimer_anfis_train_colab.ipynb`: trains ANFIS-on-GT and ANFIS-on-extracted. This notebook creates the shared split (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on the 226 subjects with complete features).
   4. `alzheimer_eval_anfis_colab.ipynb`: ANFIS metrics and confusion matrices.
   5. `alzheimer_baseline_densenet121_oasis_colab.ipynb` first, because it writes the axial slices that the other deep baselines reuse; then the VGG-16, ResNet-50, DeiT-Small and Swin-T notebooks. All of them read the split from the ANFIS-on-GT output.
3. Reproduce the tables and figures:

   ```bash
   pip install -r requirements.txt
   python scripts/evaluate.py --cohort path/to/oasis_master_features_labels.csv
   ```

   All AUCs, accuracies, sensitivities, specificities, Brier scores and DeLong p-values match Table I exactly. The bootstrap 95% CIs (5000 replicates) differ from the paper by at most 0.005, because the random seed of the original run was not recorded. Tested with Python 3.12, numpy 2.2, pandas 2.3, scipy 1.15, scikit-learn 1.8 and matplotlib 3.10.

## Protocol notes

- The ANFIS regresses the CDR score. For accuracy, sensitivity, specificity, the Brier score and the calibration curve, its output `s` is mapped to the probability `1 / (1 + exp(-(s - 0.25) / 0.5))`, a sigmoid centred on the midpoint between CDR 0 and CDR 0.5. AUC is computed on `s` directly.
- All baselines use train-fold-only age residualisation of the volumes and train-fold-only standardisation, as the ANFIS does. CART-3 uses `random_state=0`.
- The Swin-T notebook up-weights the minority class by a further factor of 1.5 and early-stops on balanced accuracy, because with plain inverse-frequency weighting it collapsed to predicting the majority class.
- Comments in the notebooks are in Chinese.

## Acknowledgment

Data were provided by OASIS. OASIS-1: Cross-Sectional: Principal Investigators: D. Marcus, R. Buckner, J. Csernansky, J. Morris; P50 AG05681, P01 AG03991, P01 AG026276, R01 AG021910, P20 MH071616, U24 RR021382.

OASIS-1 reference: D. S. Marcus et al., "Open Access Series of Imaging Studies (OASIS): Cross-sectional MRI data in young, middle aged, nondemented, and demented older adults," *J. Cogn. Neurosci.*, vol. 19, no. 9, pp. 1498–1507, 2007. <https://doi.org/10.1162/jocn.2007.19.9.1498>

## Citation

```bibtex
@misc{cao2026cognitive,
  author = {Cao, Yu and Ling, Steve},
  title  = {A Single Cognitive Score Outperforms Every {MRI}-Based Model: A Cautionary Benchmark
            for Deep and Neuro-Fuzzy Dementia Classification on {OASIS-1}},
  note   = {Manuscript},
  year   = {2026}
}
```

## License

The code and the files in `data/` and `results/` are released under the MIT License (see `LICENSE`). OASIS data are not part of this repository and remain subject to the OASIS Data Use Agreement.
