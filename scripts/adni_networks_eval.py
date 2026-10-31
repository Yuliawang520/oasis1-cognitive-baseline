"""Evaluate the five slice networks on ADNI1: families A5 to A7, increment over the MMSE, learning curve and
shortcut diagnostics (docs/prespecification_networks_adni.md).

Inputs (ADNI data; none of them is part of the repository)
  --clinical     clinical_metadata.csv of the ADNI1 "Complete 1Yr 1.5T" participants
  --adnimerge    adni_volumes_bl.csv built by scripts/adni_build_volumes.py
  --cohort-file  adni_network_cohort.csv built by scripts/adni_network_cohort.py (site, manufacturer)
  --runs-dir     the runs_adni folder written by notebooks/slice_networks_multiseed_colab.ipynb:
                 ensemble_adni_<model>_pool{full,181}.csv with columns subject_id, label, fold, seed0.., prob_demented
Output: results/adni/networks/ (aggregate tables and JSON only).

Comparisons (paired two-sided DeLong tests on the shared split, Holm within a family, paired bootstrap CIs):
  A5  each network (seed ensemble, full pool) vs LR-2                      Holm over the five networks
  A6  each network vs MMSE alone                                            Holm over the five networks
  A7  each network vs age alone and vs |age - median age|                  Holm over the ten comparisons
  increment over the MMSE: stacking of [MMSE, logit(network)] fitted on the out-of-fold scores of the training
      subjects of each fold, AUC and dAUC vs LR-1, plus the likelihood-ratio test of scripts/probe.py
  learning curve: full pool minus 181-subject pool per network, and the mean over the five networks
  diagnostics within the CDR 0 participants: Spearman rho with age and with |age - median|, Kruskal-Wallis across
      manufacturers and across sites with at least five participants of the class, within CDR 0 and within CDR > 0; LR-2, a raw-volume LR-2 (no age
      residualisation) and the MMSE serve as benchmarks

Usage: python scripts/adni_networks_eval.py --clinical ... --adnimerge ... --cohort-file ... --runs-dir ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adni_replication as ar  # noqa: E402
import evaluate as ev  # noqa: E402
import probe  # noqa: E402

NETWORKS = {"densenet121": "DenseNet-121", "vgg16": "VGG-16", "resnet50": "ResNet-50",
            "deit_small": "DeiT-Small", "swin_t": "Swin-T"}
POOLS = ("full", "181")
MIN_SITE = 5  # minimum participants of a class for a site to enter the Kruskal-Wallis test


def raw_volume_lr2(cohort: pd.DataFrame) -> np.ndarray:
    """Logistic regression on the two raw volumes, standardised on the training fold, no age residualisation."""
    x = cohort[["aseg_hippo_total", "wbv_final"]].to_numpy(dtype=float)
    y, folds = cohort["y"].to_numpy(), cohort["fold"].to_numpy()
    out = np.zeros(len(y))
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        sc = StandardScaler().fit(x[tr])
        out[te] = LogisticRegression(max_iter=1000).fit(sc.transform(x[tr]), y[tr]).predict_proba(sc.transform(x[te]))[:, 1]
    return out


def stack_oof(mmse: np.ndarray, score: np.ndarray, y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """Fusion of the MMSE with a network score: for each fold, a logistic regression on [MMSE, logit(score)] is
    fitted on the training subjects' out-of-fold network scores and applied to the test fold."""
    fused = np.zeros(len(y))
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        z_tr, z_te = np.c_[mmse[tr], probe.logit(score[tr])], np.c_[mmse[te], probe.logit(score[te])]
        sc = StandardScaler().fit(z_tr)
        fused[te] = LogisticRegression(max_iter=5000).fit(sc.transform(z_tr), y[tr]).predict_proba(sc.transform(z_te))[:, 1]
    return fused


def load_networks(runs_dir: Path, cohort: pd.DataFrame) -> tuple[dict, dict, list[str]]:
    """Ensemble and per-seed scores aligned to the cohort order. Returns (ensembles, per_seed, seed_names)."""
    ensembles, per_seed, seeds = {}, {}, None
    for name in NETWORKS:
        for pool in POOLS:
            f = runs_dir / f"ensemble_adni_{name}_pool{pool}.csv"
            if not f.exists():
                print(f"missing {f.name}")
                continue
            df = pd.read_csv(f)
            df["subject_id"] = df["subject_id"].astype(str)
            seed_cols = sorted(c for c in df.columns if c.startswith("seed"))
            seeds = seeds or seed_cols
            merged = cohort[["subject_id"]].merge(df, on="subject_id", how="left", validate="1:1")
            if merged["prob_demented"].isna().any():
                raise SystemExit(f"{f.name}: {int(merged['prob_demented'].isna().sum())} cohort participants without a score")
            ensembles[(name, pool)] = merged["prob_demented"].to_numpy()
            per_seed[(name, pool)] = {c: merged[c].to_numpy() for c in seed_cols}
    return ensembles, per_seed, seeds or []


def compare_family(y, scores_a: dict, ref: np.ndarray, n_boot: int, seed: int) -> dict:
    rows = {k: probe.compare(y, a, ref, n_boot, seed) for k, a in scores_a.items()}
    return probe.finish(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clinical", type=Path, required=True)
    ap.add_argument("--adnimerge", type=Path, required=True)
    ap.add_argument("--cohort-file", type=Path, required=True)
    ap.add_argument("--runs-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ev.REPO / "results" / "adni" / "networks")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cohort, volumes = ar.load_cohort(args.clinical, args.adnimerge)
    assert volumes, "the volume table is required"
    y, folds, age = cohort["y"].to_numpy(), cohort["fold"].to_numpy(), cohort["age"].to_numpy()
    meta = pd.read_csv(args.cohort_file)
    meta["subject_id"] = meta["subject_id"].astype(str)
    meta = cohort[["subject_id"]].merge(meta[["subject_id", "site", "manufacturer"]], on="subject_id", how="left")
    assert (meta["subject_id"].to_numpy() == cohort["subject_id"].to_numpy()).all()

    scores = ar.fit_models(cohort, True, use_anfis=False)
    scores["Age U-score"] = np.abs(age - np.median(age))
    scores["LR-2 raw"] = raw_volume_lr2(cohort)
    ensembles, per_seed, seeds = load_networks(args.runs_dir, cohort)
    full = {NETWORKS[n]: ensembles[(n, "full")] for n in NETWORKS if (n, "full") in ensembles}
    if not full:
        raise SystemExit("no full-pool ensembles found")
    nb, sd = args.n_boot, args.seed

    # model table
    models = {}
    for name, title in NETWORKS.items():
        for pool in POOLS:
            if (name, pool) not in ensembles:
                continue
            s = ensembles[(name, pool)]
            per_fold = [roc_auc_score(y[folds == k], s[folds == k]) for k in np.unique(folds)]
            seed_aucs = [roc_auc_score(y, v) for v in per_seed[(name, pool)].values()]
            models[f"{title} (pool {pool})"] = {
                "AUC": f"{roc_auc_score(y, s):.3f}", "95% CI": ev.bootstrap_ci(y, s, nb, sd),
                "Fold AUC": f"{np.mean(per_fold):.3f} ± {np.std(per_fold, ddof=1):.3f}",
                "Seed AUC": f"{np.mean(seed_aucs):.3f} ± {np.std(seed_aucs, ddof=1):.3f}" if len(seed_aucs) > 1 else f"{seed_aucs[0]:.3f}",
                "Seed range": f"{min(seed_aucs):.3f}-{max(seed_aucs):.3f}",
                "rho with age, CDR 0": f"{spearmanr(age[y == 0], s[y == 0])[0]:+.2f}",
                "rho with |age-median|, CDR 0": f"{spearmanr(scores['Age U-score'][y == 0], s[y == 0])[0]:+.2f}",
            }
    for name in ("MMSE alone", "LR-1", "LR-2", "LR-2 raw", "LR-3", "Age alone", "Age U-score"):
        s = scores[name]
        models[name] = {"AUC": f"{roc_auc_score(y, s):.3f}", "95% CI": ev.bootstrap_ci(y, s, nb, sd),
                        "rho with age, CDR 0": f"{spearmanr(age[y == 0], s[y == 0])[0]:+.2f}",
                        "rho with |age-median|, CDR 0": f"{spearmanr(scores['Age U-score'][y == 0], s[y == 0])[0]:+.2f}"}

    results = {"n": int(len(y)), "events": int(y.sum()), "seeds": seeds, "models": models}
    results["A5: networks vs LR-2"] = compare_family(y, full, scores["LR-2"], nb, sd)
    results["A6: networks vs MMSE alone"] = compare_family(y, full, scores["MMSE alone"], nb, sd)
    a7 = {f"{t} vs age alone": probe.compare(y, s, scores["Age alone"], nb, sd) for t, s in full.items()}
    a7.update({f"{t} vs |age-median|": probe.compare(y, s, scores["Age U-score"], nb, sd) for t, s in full.items()})
    results["A7: networks vs age references"] = probe.finish(a7)
    results["exploratory: networks vs LR-2 raw"] = compare_family(y, full, scores["LR-2 raw"], nb, sd)

    inc = {}
    for title, s in full.items():
        fused = stack_oof(cohort["mmse"].to_numpy(dtype=float), s, y, folds)
        row = probe.compare(y, fused, scores["LR-1"], nb, sd)
        row["p"] = ev.fmt_p(row.pop("p_raw"))
        row["AUC fused"] = f"{roc_auc_score(y, fused):.3f}"
        row["likelihood ratio"] = probe.likelihood_ratio_added(cohort["mmse"].to_numpy(dtype=float), s, y)
        inc[f"MMSE + {title} vs LR-1"] = row
    results["increment over the MMSE"] = inc

    curve = {}
    deltas = []
    for name, title in NETWORKS.items():
        if (name, "full") in ensembles and (name, "181") in ensembles:
            row = probe.compare(y, ensembles[(name, "full")], ensembles[(name, "181")], nb, sd)
            row["p"] = ev.fmt_p(row.pop("p_raw"))
            row["AUC 181"] = f"{roc_auc_score(y, ensembles[(name, '181')]):.3f}"
            curve[title] = row
            deltas.append(roc_auc_score(y, ensembles[(name, "full")]) - roc_auc_score(y, ensembles[(name, "181")]))
    if deltas:
        curve["mean dAUC over networks"] = f"{np.mean(deltas):+.3f}"
    results["learning curve: full pool minus 181"] = curve

    # scanner and site diagnostics within each class (sites with at least MIN_SITE participants of that class)
    man = meta["manufacturer"].to_numpy(dtype=object)
    site = meta["site"].astype(str).to_numpy()
    bench = {**full, "LR-2": scores["LR-2"], "LR-2 raw": scores["LR-2 raw"], "MMSE alone": scores["MMSE alone"]}
    for label, cls in (("CDR 0", y == 0), ("CDR > 0", y == 1)):
        counts = pd.Series(site[cls]).value_counts()
        big_sites = sorted(counts[counts >= MIN_SITE].index)
        diag = {}
        for title, s in bench.items():
            known = cls & pd.notna(man)
            groups = [s[known & (man == g)] for g in pd.unique(man[known])]
            kw_man = kruskal(*groups).pvalue if len(groups) > 1 else float("nan")
            site_groups = [s[cls & (site == g)] for g in big_sites]
            kw_site = kruskal(*site_groups).pvalue if len(site_groups) > 1 else float("nan")
            diag[title] = {"Kruskal-Wallis manufacturer, p": ev.fmt_p(kw_man), "manufacturers": int(len(groups)),
                           f"Kruskal-Wallis site (>= {MIN_SITE} participants), p": ev.fmt_p(kw_site), "sites": int(len(site_groups))}
        results[f"shortcut diagnostics within {label}"] = diag

    pd.DataFrame.from_dict(models, orient="index").to_csv(args.out / "adni_networks_models.csv", index_label="Model")
    (args.out / "adni_networks_results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
