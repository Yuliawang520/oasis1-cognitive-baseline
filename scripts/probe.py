"""Linear probe of frozen BrainIAC features on the shared OASIS-1 split, as an imaging-only 3D baseline.

Inputs
  --cohort    oasis_master_features_labels.csv (as for evaluate.py)
  --features  CSV written by brainiac_features.py: subject_id + 768 feature columns (not redistributed)
  --name      label of the probe in the outputs (default "BrainIAC probe")

The probe is an L2-penalised logistic regression on the standardised features, fitted on the four
training folds of the shared split; the penalty C is chosen by an inner five-fold cross-validation on the
training folds only. Everything below is pre-specified:
  * AUC on all 226 subjects and on those aged 60 or over, with bootstrap 95% CIs;
  * family F5 (Holm within the three comparisons, separately per cohort): probe vs MMSE alone,
    probe vs age alone, probe vs LR-2 (two age-residualised volumes);
  * exploratory: probe vs each deep network, and the fusion of the MMSE with the probe score
    (nested stacking) vs LR-1, with a likelihood-ratio test of adding the out-of-fold probe score to the
    MMSE in a full-cohort logistic model;
  * diagnostics of the age attribution: Spearman correlation of the probe score with age within the
    non-demented subjects, and the AUC of the probe score for separating non-demented subjects younger
    than 60 from those aged 60 or over.

Outputs: results/probe_<name>.json, results/table_probe_<name>.csv and data/oof_probe_<name>.csv
(subject_id, fold, out-of-fold probe score), which may be released like the other model outputs.

Usage: python scripts/probe.py --cohort path/to/oasis_master_features_labels.csv --features path/to/brainiac_features.csv
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as ev  # noqa: E402

C_GRID = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
# The grid specified before the features existed was 1e-4 to 10; the inner cross-validation chose its
# lower boundary in every fold, so the grid was extended downwards (primary analysis). Pass
# --c-grid 1e-4,1e-3,1e-2,1e-1,1,10 to reproduce the original grid as a sensitivity analysis.


def make_probe(c, n_pca=None):
    """Standardisation, optional PCA and L2 logistic regression, all fitted on the training data only."""
    steps = [StandardScaler()]
    if n_pca:
        steps.append(PCA(n_components=n_pca, random_state=0))
    steps.append(LogisticRegression(C=c, max_iter=5000))
    return steps


def fit_steps(steps, x, y):
    for step in steps[:-1]:
        x = step.fit_transform(x)
    steps[-1].fit(x, y)
    return steps


def predict_steps(steps, x):
    for step in steps[:-1]:
        x = step.transform(x)
    return steps[-1].predict_proba(x)[:, 1]


def choose_c(x, y, n_pca, seed, c_grid=C_GRID):
    """Inner five-fold CV on the training folds; returns the C with the highest mean AUC."""
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    means = {}
    for c in c_grid:
        aucs = []
        for tr, va in inner.split(x, y):
            steps = fit_steps(make_probe(c, n_pca), x[tr], y[tr])
            aucs.append(roc_auc_score(y[va], predict_steps(steps, x[va])))
        means[c] = float(np.mean(aucs))
    return max(means, key=means.get), means


def probe_oof(x, y, folds, n_pca, seed, c_grid=C_GRID):
    oof, chosen = np.zeros(len(y)), {}
    for k in np.unique(folds):
        train, test = folds != k, folds == k
        c, _ = choose_c(x[train], y[train], n_pca, seed, c_grid)
        chosen[int(k)] = c
        steps = fit_steps(make_probe(c, n_pca), x[train], y[train])
        oof[test] = predict_steps(steps, x[test])
    return oof, chosen


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fusion_oof(x, mmse, y, folds, chosen, n_pca, oof_probe):
    """Nested stacking: for each outer fold, the training subjects receive probe scores from models that
    did not see them (inner leave-one-training-fold-out), the fusion logistic regression on
    [MMSE, logit(probe score)] is fitted on those, and the test fold uses the outer probe scores."""
    fused = np.zeros(len(y))
    for k in np.unique(folds):
        train, test = folds != k, folds == k
        inner_scores = np.zeros(len(y))
        for j in np.unique(folds[train]):
            fit_mask = train & (folds != j)
            steps = fit_steps(make_probe(chosen[int(k)], n_pca), x[fit_mask], y[fit_mask])
            inner_scores[folds == j] = predict_steps(steps, x[folds == j])
        z_train = np.c_[mmse[train], logit(inner_scores[train])]
        z_test = np.c_[mmse[test], logit(oof_probe[test])]
        scaler = StandardScaler().fit(z_train)
        model = LogisticRegression(max_iter=5000).fit(scaler.transform(z_train), y[train])
        fused[test] = model.predict_proba(scaler.transform(z_test))[:, 1]
    return fused


def likelihood_ratio_added(mmse, score, y):
    """Full-cohort test of adding the (out-of-fold) probe score to the MMSE."""
    def loglik(cols):
        z = (cols - cols.mean(0)) / cols.std(0, ddof=0)
        m = LogisticRegression(penalty=None, max_iter=5000).fit(z, y)
        p = np.clip(m.predict_proba(z)[:, 1], 1e-12, 1 - 1e-12)
        return np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    stat = 2 * (loglik(np.c_[mmse, logit(score)]) - loglik(mmse[:, None]))
    return {"chi2": f"{stat:.1f}", "p": ev.fmt_p(chi2.sf(stat, 1))}


def compare(y, a, b, n_boot, seed):
    """a minus b: AUC difference, paired bootstrap CI and two-sided DeLong p."""
    return {"dAUC": f"{roc_auc_score(y, a) - roc_auc_score(y, b):+.3f}",
            "95% CI": ev.bootstrap_delta_ci(y, a, b, n_boot, seed), "p_raw": ev.delong_p(y, a, b)}


def finish(rows):
    """Format p values and add Holm-adjusted p values within the given rows."""
    ps = [r["p_raw"] for r in rows.values()]
    for r, h in zip(rows.values(), ev.holm(ps)):
        r["p"], r["Holm p"] = ev.fmt_p(r.pop("p_raw")), ev.fmt_p(h)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cohort", type=Path, required=True)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--name", default="BrainIAC probe")
    ap.add_argument("--pca", type=int, default=None, help="optional PCA dimension before the logistic regression")
    ap.add_argument("--c-grid", default=None, help="comma-separated penalty grid (default: 1e-6 to 10)")
    ap.add_argument("--data", type=Path, default=ev.REPO / "data")
    ap.add_argument("--out", type=Path, default=ev.REPO / "results")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    slug = re.sub(r"[^a-z0-9]+", "_", args.name.lower()).strip("_")

    cohort, y, folds, scores, _probs = ev.load(args.cohort, args.data)
    feats = pd.read_csv(args.features)
    feats = cohort[["subject_id"]].merge(feats, on="subject_id", how="left", validate="1:1")
    have = feats.drop(columns="subject_id").notna().all(axis=1).to_numpy()
    if not have.all():
        print(f"WARNING: features for only {have.sum()}/{len(have)} subjects; restricting all analyses to them")
    x = feats.drop(columns="subject_id").to_numpy(dtype=float)[have]
    y, folds, cohort = y[have], folds[have], cohort[have].reset_index(drop=True)
    scores = {k: v[have] for k, v in scores.items()}
    mmse, age = cohort["mmse"].to_numpy(dtype=float), cohort["age"].to_numpy(dtype=float)
    print(f"{args.name}: {x.shape[1]} features, {len(y)} subjects, {y.sum()} demented")

    c_grid = tuple(float(c) for c in args.c_grid.split(",")) if args.c_grid else C_GRID
    oof, chosen = probe_oof(x, y, folds, args.pca, args.seed, c_grid)
    fused = fusion_oof(x, mmse, y, folds, chosen, args.pca, oof)
    scores[args.name], scores["MMSE + " + args.name] = oof, fused
    print("penalty C per fold:", chosen)

    old = age >= 60
    results = {"name": args.name, "n_features": int(x.shape[1]), "pca": args.pca, "C_grid": list(c_grid),
               "C_per_fold": chosen, "n": int(len(y)), "events": int(y.sum())}
    table = {}
    for label, mask in (("all", np.ones(len(y), bool)), ("60+", old)):
        yy, po, fu = y[mask], oof[mask], fused[mask]
        sub = {k: v[mask] for k, v in scores.items()}
        per_fold = [roc_auc_score(yy[folds[mask] == k], po[folds[mask] == k]) for k in np.unique(folds)]
        results[f"AUC ({label})"] = {"probe": f"{roc_auc_score(yy, po):.3f}", "95% CI": ev.bootstrap_ci(yy, po, args.n_boot, args.seed),
                                     "fold AUC": f"{np.mean(per_fold):.3f} ± {np.std(per_fold, ddof=1):.3f}",
                                     "fusion": f"{roc_auc_score(yy, fu):.3f}"}
        f5 = {other: compare(yy, po, sub[other], args.n_boot, args.seed) for other in ("MMSE alone", "Age alone", "LR-2")}
        results[f"F5 ({label})"] = finish(f5)
        deep = {m: compare(yy, po, sub[m], args.n_boot, args.seed) for m in ev.DEEP_COLUMNS}
        results[f"probe vs deep networks ({label}, exploratory)"] = finish(deep)
        fusion = {"vs LR-1": compare(yy, fu, sub["LR-1"], args.n_boot, args.seed),
                  "vs MMSE alone": compare(yy, fu, sub["MMSE alone"], args.n_boot, args.seed),
                  "vs LR-3": compare(yy, fu, sub["LR-3"], args.n_boot, args.seed)}
        for r in fusion.values():
            r["p"] = ev.fmt_p(r.pop("p_raw"))
        results[f"fusion ({label}, exploratory)"] = fusion
        for other, r in f5.items():
            table[f"{label}: probe vs {other}"] = r
    results["likelihood ratio, MMSE + probe vs MMSE (all)"] = likelihood_ratio_added(mmse, oof, y)
    non_demented = y == 0
    rho = spearmanr(age[non_demented], oof[non_demented])[0]
    results["diagnostics"] = {
        "rho with age, non-demented": f"{rho:+.2f}",
        "AUC young vs old among non-demented": f"{roc_auc_score(old[non_demented].astype(int), oof[non_demented]):.3f}",
        "rho with age, non-demented, fusion": f"{spearmanr(age[non_demented], fused[non_demented])[0]:+.2f}",
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"probe_{slug}.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    pd.DataFrame.from_dict(table, orient="index").to_csv(args.out / f"table_probe_{slug}.csv", index_label="Comparison")
    pd.DataFrame({"subject_id": cohort["subject_id"], "fold": folds, slug: oof,
                  f"mmse_plus_{slug}": fused}).to_csv(args.data / f"oof_probe_{slug}.csv", index=False, float_format="%.10g")
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
