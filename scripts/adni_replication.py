"""External replication of the tabular attribution on the ADNI1 baseline cohort.

Inputs (ADNI data cannot be redistributed; neither file is part of the repository)
  --clinical   clinical_metadata.csv of the ADNI1 "Complete 1Yr 1.5T" subjects: subject_id (PTID),
               timepoint, age_exact, mmscore, cdglobal (built from the IDA metadata by adni_08_parse_ida_metadata.py)
  --adnimerge  ADNIMERGE.csv from the IDA (optional): Hippocampus, WholeBrain and ICV at VISCODE 'bl',
               merged by PTID. Without it only the zero-parameter references and LR-1 are evaluated.

The label is the same operational label as on OASIS-1, a global CDR above 0 (in ADNI1 this covers the
MCI and AD groups, so the positive class is cognitive impairment rather than dementia). The split is a
stratified five-fold split with random_state 42, as on OASIS-1. The models, the train-fold-only age
residualisation, the ANFIS settings and the statistics are those of scripts/evaluate.py and
scripts/robustness.py.

Comparison families (Holm within each family):
  A1  MMSE alone vs each model with imaging inputs (LR-3, CART-3, ANFIS, LR-2)
  A2  LR-3 vs LR-1, with the likelihood-ratio test of the two volume coefficients
  A3  LR-3 vs ANFIS on identical inputs
  A4  age alone vs each imaging-only model (LR-2)
Diagnostic: Spearman correlation of every model output with age within the CDR 0 subjects; the zero-parameter
'Age U-score' (|age - median age|) measures the non-monotonic age information of the cohort.

Usage: python scripts/adni_replication.py --clinical path/to/clinical_metadata.csv [--adnimerge path/to/ADNIMERGE.csv] [--anfis]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as ev  # noqa: E402

THREE = {"hippo": "aseg_hippo_total", "wbv": "wbv_final", "mmse": "mmse"}


def load_cohort(clinical_csv, adnimerge_csv):
    c = pd.read_csv(clinical_csv)
    c = c[c["timepoint"] == "bl"].dropna(subset=["age_exact", "mmscore", "cdglobal"])
    cohort = pd.DataFrame({"subject_id": c["subject_id"].astype(str).str.strip(), "age": c["age_exact"].astype(float),
                           "mmse": c["mmscore"].astype(float), "cdr_numeric": c["cdglobal"].astype(float)})
    cohort["y"] = (cohort["cdr_numeric"] > 0).astype(int)
    volumes = False
    if adnimerge_csv:
        m = pd.read_csv(adnimerge_csv, low_memory=False)
        m = m[m["VISCODE"].astype(str).str.lower() == "bl"][["PTID", "Hippocampus", "WholeBrain", "ICV", "FSVERSION"]]
        m = m.rename(columns={"PTID": "subject_id"}).drop_duplicates("subject_id")
        n_before = len(cohort)
        cohort = cohort.merge(m, on="subject_id", how="left", validate="1:1")
        cohort["aseg_hippo_total"] = pd.to_numeric(cohort["Hippocampus"], errors="coerce")
        cohort["wbv_final"] = pd.to_numeric(cohort["WholeBrain"], errors="coerce") / pd.to_numeric(cohort["ICV"], errors="coerce")
        missing = cohort[["aseg_hippo_total", "wbv_final"]].isna().any(axis=1)
        print(f"ADNIMERGE: baseline volumes for {(~missing).sum()}/{n_before} subjects; "
              f"FreeSurfer versions {cohort['FSVERSION'].dropna().unique().tolist()}")
        cohort = cohort[~missing]
        volumes = True
    cohort = cohort.reset_index(drop=True)
    folds = np.zeros(len(cohort), dtype=int)
    for k, (_, test) in enumerate(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(cohort, cohort["y"])):
        folds[test] = k
    cohort["fold"] = folds
    return cohort, volumes


def fit_models(cohort, volumes, use_anfis):
    logistic = lambda: LogisticRegression(max_iter=1000)
    tree = lambda: DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=0)
    scores = {"MMSE alone": ((30 - cohort["mmse"]) / 30).to_numpy(),
              "Age alone": cohort["age"].to_numpy(dtype=float),
              # non-monotonic age reference: distance from the median age (ADNI1 enrols few controls under 70)
              "Age U-score": np.abs(cohort["age"].to_numpy(dtype=float) - np.median(cohort["age"].to_numpy(dtype=float))),
              "LR-1": ev.out_of_fold(cohort, {"mmse": "mmse"}, logistic)}
    if volumes:
        scores["LR-3"] = ev.out_of_fold(cohort, THREE, logistic)
        scores["CART-3"] = ev.out_of_fold(cohort, THREE, tree)
        scores["LR-2"] = ev.out_of_fold(cohort, {"hippo": "aseg_hippo_total", "wbv": "wbv_final"}, logistic)
        if use_anfis:
            from robustness import anfis_oof  # noqa: E402  (needs torch and imbalanced-learn)
            from anfis import Config  # noqa: E402
            scores["ANFIS"] = anfis_oof(cohort, cohort["fold"].to_numpy(), ("hippo", "wbv", "mmse"), Config())
    return scores


def compare(y, a, b, n_boot, seed):
    return {"dAUC": f"{roc_auc_score(y, a) - roc_auc_score(y, b):+.3f}",
            "95% CI": ev.bootstrap_delta_ci(y, a, b, n_boot, seed), "p_raw": ev.delong_p(y, a, b)}


def finish(rows):
    ps = [r["p_raw"] for r in rows.values()]
    for r, h in zip(rows.values(), ev.holm(ps)):
        r["p"], r["Holm p"] = ev.fmt_p(r.pop("p_raw")), ev.fmt_p(h)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clinical", type=Path, required=True)
    ap.add_argument("--adnimerge", type=Path, default=None)
    ap.add_argument("--anfis", action="store_true", help="also fit the paper's ANFIS (needs torch and imbalanced-learn)")
    ap.add_argument("--out", type=Path, default=ev.REPO / "results" / "adni")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cohort, volumes = load_cohort(args.clinical, args.adnimerge)
    y, folds = cohort["y"].to_numpy(), cohort["fold"].to_numpy()
    age = cohort["age"].to_numpy()
    print(f"ADNI1 baseline: n = {len(y)}, CDR > 0 = {y.sum()} ({100 * y.mean():.0f}%), "
          f"age {age.mean():.1f} ± {age.std(ddof=1):.1f} (range {age.min():.0f}-{age.max():.0f}); "
          f"CDR 0 age {age[y == 0].mean():.1f}, CDR > 0 age {age[y == 1].mean():.1f}")
    scores = fit_models(cohort, volumes, args.anfis)

    models = {}
    for name, s in scores.items():
        per_fold = [roc_auc_score(y[folds == k], s[folds == k]) for k in np.unique(folds)]
        models[name] = {"AUC": f"{roc_auc_score(y, s):.3f}", "95% CI": ev.bootstrap_ci(y, s, args.n_boot, args.seed),
                        "Fold AUC": f"{np.mean(per_fold):.3f} ± {np.std(per_fold, ddof=1):.3f}",
                        "rho with age, CDR 0": f"{spearmanr(age[y == 0], s[y == 0])[0]:+.2f}"}
    under70 = age < 70
    results = {"n": int(len(y)), "events": int(y.sum()), "volumes": volumes, "models": models,
               "age structure": {"median age": f"{np.median(age):.1f}",
                                 "age SD, CDR 0": f"{age[y == 0].std(ddof=1):.1f}", "age SD, CDR > 0": f"{age[y == 1].std(ddof=1):.1f}",
                                 "under 70: n": int(under70.sum()), "under 70: CDR > 0": int(y[under70].sum()),
                                 "80 or over: n": int((age >= 80).sum()), "80 or over: CDR > 0": int(y[age >= 80].sum())}}

    if volumes:
        imaging = [m for m in ("LR-3", "CART-3", "ANFIS", "LR-2") if m in scores]
        results["A1: MMSE alone vs models with imaging inputs"] = finish(
            {m: compare(y, scores[m], scores["MMSE alone"], args.n_boot, args.seed) for m in imaging})
        a2 = compare(y, scores["LR-3"], scores["LR-1"], args.n_boot, args.seed)
        a2["p"] = ev.fmt_p(a2.pop("p_raw"))
        a2["likelihood ratio"] = ev.likelihood_ratio(cohort, np.ones(len(y), bool))
        results["A2: LR-3 vs LR-1"] = a2
        if "ANFIS" in scores:
            a3 = compare(y, scores["LR-3"], scores["ANFIS"], args.n_boot, args.seed)
            a3["p"] = ev.fmt_p(a3.pop("p_raw"))
            results["A3: LR-3 vs ANFIS"] = a3
        results["A4: age alone vs imaging-only models"] = finish(
            {m: compare(y, scores[m], scores["Age alone"], args.n_boot, args.seed) for m in ("LR-2",)})
    results["exploratory: LR-1 vs MMSE alone"] = {**compare(y, scores["LR-1"], scores["MMSE alone"], args.n_boot, args.seed)}
    results["exploratory: LR-1 vs MMSE alone"]["p"] = ev.fmt_p(results["exploratory: LR-1 vs MMSE alone"].pop("p_raw"))

    pd.DataFrame.from_dict(models, orient="index").to_csv(args.out / "adni_models.csv", index_label="Model")
    (args.out / "adni_results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    pd.DataFrame({"subject_id": cohort["subject_id"], "fold": folds}).to_csv(args.out / "adni_folds.csv", index=False)
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
