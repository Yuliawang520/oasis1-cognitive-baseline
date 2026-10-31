# Pre-specified analysis plan: slice networks on ADNI1 and seed sensitivity on OASIS-1

Committed on 2026-11-01, before any network was trained on ADNI1 or re-trained on OASIS-1 with a fixed seed. The
plan fixes the data, the pipeline, the comparisons and the reading of each possible outcome. Deviations will be
listed at the end of this file with their date and reason.

## Questions

- **Q1 (ADNI1, primary).** On the 601 ADNI1 participants of the tabular replication (Table VI of the manuscript), do
  the five slice-based networks exceed the two-volume logistic regression LR-2 on the shared split, and what do they
  add to the MMSE reference?
- **Q2 (ADNI1, learning curve).** How much of the networks' accuracy depends on the size of the training pool
  (about 481 subjects per fold on ADNI1 against 181 on OASIS-1)?
- **Q3 (OASIS-1, seed sensitivity).** How much do the AUCs of the five networks vary across training seeds, and do
  the conclusions of families F1, F4 and F5 hold for every seed?

The age-only reference carries no monotonic information on ADNI1 (AUC 0.470), so Q1 does not replicate the OASIS-1
age attribution. ADNI1 has instead a non-monotonic age structure: 96 of the 104 participants under 70 have a CDR
above 0, and the score |age − median age| reaches an AUC of 0.637. The shortcut diagnostics below therefore use both
age and |age − median|.

## Data and split

- ADNI1: the same 601 participants and the same five folds (StratifiedKFold, shuffle, random state 42) as the
  tabular replication; label: global CDR > 0 at baseline (410 participants); image: the baseline 1.5 T T1 of the
  "ADNI1: Complete 1Yr 1.5T" collection in ADNI's preprocessed form (GradWarp, B1 correction where acquired, N3,
  Scaled), one image per participant chosen by `adni_01_build_manifest.py` of the longitudinal project (first
  pipeline run, B1-corrected and GradWarped versions preferred, MPR before MPR-R).
- OASIS-1: the 226 subjects, the shared split and the existing slice PNGs of the original notebooks.
- Scanner manufacturer and model (ADNIMERGE2 MRIQC, screening MPRAGE series) and site (first three characters of
  the participant ID) are recorded for the diagnostics.

## Pipeline

Identical to the OASIS-1 notebooks except where stated (`scripts/slice_networks.py` re-implements them; the
regression test in the commit message reproduces the OASIS-1 PNGs to within one grey level).

- **ADNI1 slices** (`scripts/adni_slices.py`, `notebooks/adni_slices_colab.ipynb`): HD-BET 2 with test-time
  augmentation on the RAS-reoriented T1 → FreeSurfer-style conform (1 mm, 256³, LIA, trilinear, centred on the
  scanner field of view like `mri_convert --conform`) → white-matter peak scaled to 110 and clipped to 0–255 (the
  T1.mgz convention behind OASIS-1's brainmask.mgz) → the same 24 coronal slices (indices 69 to 119 of the third
  axis) → 8-bit PNG by per-slice min-max scaling. The original notebooks call these slices "axial"; they are
  coronal (the third axis of a conformed FreeSurfer volume runs posterior to anterior), and the manuscript is
  corrected accordingly.
  - QC: fraction of the 24 slice positions that lie inside the brain mask, white-matter peak, mask volume, and a
    montage. **Fallback rule:** if the slice window leaves the brain in more than 10% of participants, the
    brain-centred variant (`--center brain`: integer shift of the conformed grid so that the brain's bounding box
    is centred, no resampling) is used for all participants and reported as such.
- **Networks:** DenseNet-121, VGG-16, ResNet-50, DeiT-Small, Swin-T with the original hyperparameters (ImageNet
  weights, two phases of 12 and 15 epochs, AdamW, cosine schedules, early stopping with patience 6 on the raw
  validation accuracy, balanced accuracy for Swin-T, inverse-frequency class weights with the Swin-T minority boost
  of 1.5, AMP, gradient clipping, batch size 32, 85/15 subject-level validation split by md5 hash, subject-level
  mean of the slice probabilities). New: seeds 0, 1 and 2 (random, numpy, torch; deterministic cuDNN; four data
  loader workers). A few CUDA kernels remain non-deterministic, so a repeated run of one seed can differ in the
  third decimal of the AUC.
- **Learning curve (Q2):** for each fold, the training pool (training plus validation subjects) is subsampled with
  a seeded generator, stratified by label, to 181 subjects, the mean OASIS-1 pool; the test fold is unchanged.
- **Primary output per (model, pool):** the mean of the three seeds' out-of-fold probabilities (seed ensemble).
  The per-seed AUCs are reported as mean ± SD and range.

## Statistics

All tests are paired two-sided DeLong tests on the shared split with Holm correction within a family, 95% CIs of
differences from 5000 paired bootstrap replicates (seed 0), as in the manuscript.

- **A5 (primary, Q1):** each network's seed ensemble (full pool) versus LR-2; Holm over the five networks.
- **A6:** each network versus the MMSE reference. **A7:** each network versus age alone and versus
  |age − median| (two references, Holm over ten comparisons).
- **Increment over the MMSE:** stacking of the MMSE with each network's ensemble output: for each fold, a logistic
  regression on [MMSE, logit(network score)] is fitted on the out-of-fold scores of the training subjects and applied
  to the test fold (the probe's inner refit is not possible for a trained network), plus the likelihood-ratio test
  of `scripts/probe.py` and ΔAUC against LR-1 with its CI.
- **Learning curve (Q2):** ΔAUC (full pool minus 181) per network with a paired bootstrap CI; the five networks'
  mean ΔAUC is the summary statistic.
- **Shortcut diagnostics** within the 191 CDR 0 participants: Spearman ρ of the ensemble output with age and with
  |age − median|; Kruskal–Wallis test of the output across manufacturers and across sites with at least ten
  participants. The within-CDR 0 ρ of a logistic regression on the two raw (non-residualised) volumes, 0.38, is
  the benchmark for "atrophy that correlates with age".
- **Q3 (OASIS-1 seeds):** per-seed AUC, mean ± SD and range for each network; whether the original unseeded run
  lies inside the seed range; families F1 (versus MMSE), F4 (versus age alone, all subjects and aged 60 or over)
  and the probe comparisons of F5 re-evaluated with the seed ensemble. A conclusion is "stable" if its direction
  and its significance hold for the ensemble and for each seed.
- **Decision rule for the manuscript:** an original OASIS-1 network run stays the primary result if it lies within
  the seed range of its network; otherwise the seed ensemble replaces it and all dependent numbers are regenerated.

## Readings fixed in advance

- Networks do not exceed LR-2 on ADNI1 (A5 not significant, or networks below) **and** the learning curve is flat
  (mean ΔAUC below 0.02): the two-volume logistic regression is not exceeded by this family of networks at either
  sample size; the OASIS-1 result is not a sample-size artefact.
- Networks exceed LR-2 at the full pool but not at 181: the OASIS-1 network results reflect training-set size in
  part; the manuscript narrows its statement about networks to small cohorts, and the increment over the MMSE
  decides the clinical reading.
- Networks exceed LR-2 at both pool sizes: the slice networks extract more than two volumes on ADNI1, and the
  OASIS-1 failure is cohort-specific (age structure, preprocessing or label mix); reported as such.
- ρ with age within CDR 0 near the raw-volume benchmark (about 0.3 to 0.4) is consistent with atrophy that
  correlates with age; values well above it, or a strong association with |age − median|, indicate an age shortcut.
  A manufacturer or site effect within CDR 0 is reported as a scanner shortcut and discussed, not corrected.

## Excluded or not planned

- No hyperparameter tuning on ADNI1; no 3D networks; the BrainIAC probe is not evaluated on ADNI1 (its pretraining
  pool contains ADNI scans).
- Per-participant ADNI predictions are not released (ADNI Data Use Agreement); the repository receives aggregate
  tables only.

## Deviations

(none yet)
