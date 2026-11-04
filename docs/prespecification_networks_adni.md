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
- Scanner manufacturer and model (the IDA metadata XML of the baseline image itself; the ADNIMERGE2 MRIQC
  table, screening MPRAGE series, only for images without an XML) and site (first three characters of the
  participant ID) are recorded for the diagnostics.

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
  |age − median|; Kruskal–Wallis test of the output across manufacturers and across sites with at least five
  participants of the class under test (within CDR 0 and, separately, within CDR > 0). The within-CDR 0 ρ of a logistic regression on the two raw (non-residualised) volumes, 0.38, is
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

The first two were recorded before any network output existed; the third after the results.

- 2026-11-02, scanner metadata. The first cohort file took manufacturer and model from the ADNIMERGE2 MRIQC
  table (most frequent value over a participant's screening MPRAGE series). That left 136 of the 601 participants
  without a manufacturer and, for 9 participants who also took part in the ADNI1 3 T sub-study, named the 3 T
  scanner rather than the 1.5 T image used here. The cohort file now takes both from the IDA metadata XML of the
  exact baseline image (601 of 601 matched; GE 300, Siemens 236, Philips 65; MRIQC kept as fallback). Folds,
  labels and image paths are unchanged. The field strength and the in-plane pixel spacing (0.94 mm for 350
  participants, 1.25 mm for 220) are stored for information only; no test on them is pre-specified.
- 2026-11-02, documentation of the site threshold. The text above said "sites with at least ten participants";
  the tagged code (`scripts/adni_networks_eval.py`, `MIN_SITE = 5`) has used five participants of the class
  under test since the pipeline test on synthetic runs, because no site has ten CDR 0 participants. The text
  now states what the code does; the code is unchanged.
- 2026-11-02, reported network runs on OASIS-1 (after the results). The rule above kept a single run unless its
  AUC lay outside the three-seed range. Three of the five single runs lay outside, by 0.004 (DeiT-Small, above),
  0.007 (VGG-16, below) and 0.001 (ResNet-50, above); a literal application would mix single runs and ensembles
  in Table II. We replaced all five single runs by the seed ensembles (`evaluate.py --network-runs ensemble`, now
  the default). For DenseNet-121 and Swin-T, whose single runs lay inside their ranges, this raised the AUC by
  0.015 and 0.014. Consequences: no F1 conclusion changed; in F4, VGG-16 became significant on all subjects (the
  single run had Holm p 0.078, every seed and the ensemble p ≤ 0.036), and in the subjects aged 60 or over all
  five ensembles exceeded age alone (ResNet-50, DenseNet-121 and Swin-T did so in 0, 2 and 1 of 3 seeds). The
  manuscript therefore reports the ensemble results and states which conclusions hold in every seed (Table S8).

## Checkpoints

- 2026-11-02, slice QC on Colab (A100, HD-BET with test-time augmentation): 601 of 601 participants produced
  24 slices; the FOV-centred window 69–119 lay inside the brain for every participant (0 %, fallback threshold
  10 %), so the FOV-centred slices remain the primary input. In the six-participant preview the window covered
  25–55 % of each brain's anterior–posterior extent, against 23–53 % in 40 local OASIS-1 volumes. All 601 source
  images are float32 with continuous values (no integer quantisation), so the white-matter normalisation is safe
  for the low-intensity-scale scanners (raw white-matter peak 33–2952).
- 2026-11-02, Q3 (OASIS-1 seeds, A100, 3.1 GPU hours). Seed means lay within 0.016 of the single runs; seed SD
  0.001 (ResNet-50) to 0.047 (Swin-T); the architecture ranking changed with the seed. F1 stable for every
  network and seed. F4: DeiT-Small and VGG-16 above age alone in every seed in both cohorts; ResNet-50,
  DenseNet-121 and Swin-T crossed the Holm threshold between seeds. Swin-T seeds 1 and 2 stopped at epochs 1–3 in
  several folds (balanced-accuracy early stopping on 24 validation subjects), so their pooled AUC lay 0.05–0.10
  below the mean fold AUC.
- 2026-11-02, Q1 and Q2 (ADNI1, A100, 10.3 GPU hours, 150 fold runs). Seed ensembles: DeiT-Small 0.681, VGG-16
  0.674, Swin-T 0.666, DenseNet-121 0.653, ResNet-50 0.633. A5: all five below LR-2 (0.779) by 0.098–0.146, Holm
  p ≤ 0.0001. A6: below the MMSE score by 0.200–0.248. A7: above age alone by 0.163–0.211; none above
  |age − median| (−0.004 to +0.044, Holm p ≥ 0.90). Learning curve: full pool minus 181 participants +0.007 to
  +0.022, mean +0.016, none significant. Increment over the MMSE: +0.007 to +0.015 (volumes: +0.031). ρ with age
  within CDR 0: 0.27–0.44 (raw-volume benchmark 0.38); ρ with |age − median| ≤ 0.05. Kruskal–Wallis within CDR 0:
  no manufacturer effect (p ≥ 0.15); one site effect of five (ResNet-50, p = 0.028). Seed SD 0.001–0.020. The
  first pre-specified reading applies: the two-volume logistic regression is not exceeded by this family of
  networks at either sample size, and the OASIS-1 result is not a sample-size artefact.
- 2026-11-04, diagnostics added before submission (exploratory, no new families, no change to A1–A7 or F1–F5):
  a fitted age reference (out-of-fold logistic regression on centred age and its square; OASIS-1 0.692, ADNI1
  0.630), the age-adjusted AUC of Janes and Pepe within five-year age strata (OASIS-1: networks 0.615–0.674 against
  pooled 0.691–0.759, LR-2 0.805, MMSE 0.880; ADNI1: networks 0.678–0.720, LR-2 0.790), the CN-versus-MCI
  restriction on ADNI1 (MMSE alone 0.836, LR-2 0.743), 100 repeated splits of the ADNI1 tabular models (MMSE above
  LR-2 and LR-3 above LR-1 in every split) and calibration statistics (Tables S10 and S11 of the manuscript).
