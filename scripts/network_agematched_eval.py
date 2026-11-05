"""Age-matched training on OASIS-1 (Q4 of docs/prespecification_networks_adni.md): the five networks retrained on
the 189 subjects aged 60 or over (shared folds restricted to them, seeds 0, 1 and 2, seed ensemble), compared on
the same subjects with the networks trained on all 226 subjects, with age alone and with LR-2, plus the BrainIAC
linear probe refitted on the same subjects.

Inputs
  --cohort    oasis_master_features_labels.csv (ages and labels, as for evaluate.py)
  --data      data/ with the released split and predictions (default: the data/ folder of the repository); the
              matched-trained predictions are read from data/oof_predictions_agematched.csv (columns subject_id,
              fold, <model>_seed0.., <model>_ensemble; no labels)
  --runs-dir  optional: read them instead from the runs_oasis60 folder written by
              notebooks/slice_networks_multiseed_colab.ipynb with COHORT = 'oasis60'
              (ensemble_oasis60_<model>_poolfull.csv; the released file holds the same numbers)
  --features  brainiac_features.csv (optional): refits the probe on the matched subjects

Families (paired two-sided DeLong tests on the 189 subjects, paired bootstrap 95% CIs, Holm within each family):
  M1  matched-trained ensemble vs the all-trained ensemble on the same subjects   (five comparisons; primary)
  M2  matched-trained ensemble vs age alone                                          (five)
  M3  matched-trained ensemble vs LR-2                                               (five)
Diagnostics: per-seed AUCs, per-fold AUCs (mean ± SD over the five test folds), Spearman rho with age within the
non-demented subjects and the age-adjusted AUC (five-year strata) for both training variants; the refitted probe
vs the all-fitted probe (exploratory), with pooled and per-fold AUCs, because the penalties chosen by the inner
cross-validation can shrink the probe scores to fold-specific scales, which lowers the pooled AUC but not the
fold AUCs.

Outputs: results/table_s12_agematched.csv and results/agematched.json (aggregate only).
Usage: python scripts/network_agematched_eval.py --cohort ... [--runs-dir path/to/runs_oasis60] [--features ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as ev  # noqa: E402
import probe  # noqa: E402

RUN_NAMES = {"DeiT-Small": "deit_small", "VGG-16": "vgg16", "ResNet-50": "resnet50",
             "DenseNet-121": "densenet121", "Swin-T": "swin_t"}
MIN_AGE = 60
COHORT_NAME = "oasis60"
AGEMATCHED_FILE = "oof_predictions_agematched.csv"  # released matched-trained predictions (no labels)


def load_runs(runs_dir: Path | None, data_dir: Path, ids: pd.Series) -> tuple[dict, dict, list[str]]:
    """Per-seed and ensemble scores of each matched-trained network, from the released data file (default) or
    from the Colab run folder, aligned to the matched subjects."""
    ensembles, per_seed, seeds = {}, {}, []
    key = pd.DataFrame({"subject_id": ids.astype(str).to_numpy()})
    if runs_dir is None:
        df = pd.read_csv(data_dir / AGEMATCHED_FILE)
        df["subject_id"] = df["subject_id"].astype(str)
        merged = key.merge(df, on="subject_id", how="left", validate="1:1")
        for title, name in RUN_NAMES.items():
            seed_cols = sorted(c for c in df.columns if c.startswith(f"{name}_seed"))
            if not seed_cols:
                continue
            if merged[f"{name}_ensemble"].isna().any():
                raise SystemExit(f"{AGEMATCHED_FILE}: matched subjects without a score for {title}")
            seeds = seeds or [c.split("_")[-1] for c in seed_cols]
            ensembles[title] = merged[f"{name}_ensemble"].to_numpy()
            per_seed[title] = {c.split("_")[-1]: merged[c].to_numpy() for c in seed_cols}
        return ensembles, per_seed, seeds
    for title, name in RUN_NAMES.items():
        f = runs_dir / f"ensemble_{COHORT_NAME}_{name}_poolfull.csv"
        if not f.exists():
            print(f"missing {f.name}")
            continue
        df = pd.read_csv(f)
        df["subject_id"] = df["subject_id"].astype(str)
        merged = key.merge(df, on="subject_id", how="left", validate="1:1")
        if merged["prob_demented"].isna().any():
            raise SystemExit(f"{f.name}: {int(merged['prob_demented'].isna().sum())} matched subjects without a score")
        seed_cols = sorted(c for c in df.columns if c.startswith("seed"))
        seeds = seeds or seed_cols
        ensembles[title] = merged["prob_demented"].to_numpy()
        per_seed[title] = {c: merged[c].to_numpy() for c in seed_cols}
    return ensembles, per_seed, seeds


def fold_auc(y, s, folds) -> tuple[str, list[float]]:
    per = [roc_auc_score(y[folds == k], s[folds == k]) for k in np.unique(folds)]
    return f"{np.mean(per):.3f} ± {np.std(per, ddof=1):.3f}", [round(float(v), 3) for v in per]


def family(y, a_scores: dict, ref, n_boot, seed):
    rows = {k: probe.compare(y, a, ref, n_boot, seed) for k, a in a_scores.items()}
    return probe.finish(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cohort", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=ev.REPO / "data")
    ap.add_argument("--runs-dir", type=Path, default=None)
    ap.add_argument("--features", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=ev.REPO / "results")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    nb, sd = args.n_boot, args.seed

    cohort, y, folds, scores, _, _ = ev.load(args.cohort, args.data)
    age = cohort["age"].to_numpy(dtype=float)
    old = age >= MIN_AGE
    yy, age60, folds60 = y[old], age[old], folds[old]
    ids = cohort["subject_id"][old]
    matched, per_seed, seeds = load_runs(args.runs_dir, args.data, ids)
    if not matched:
        raise SystemExit("no matched-trained ensembles found")
    all_trained = {t: scores[t][old] for t in matched}
    non_dem = yy == 0

    table = {}
    for t, s in matched.items():
        seed_aucs = [roc_auc_score(yy, v) for v in per_seed[t].values()]
        table[t] = {"AUC matched-trained": f"{roc_auc_score(yy, s):.3f}", "95% CI": ev.bootstrap_ci(yy, s, nb, sd),
                    "Fold AUC matched-trained": fold_auc(yy, s, folds60)[0],
                    "Seed AUC": f"{np.mean(seed_aucs):.3f} ± {np.std(seed_aucs, ddof=1):.3f}" if len(seed_aucs) > 1 else f"{seed_aucs[0]:.3f}",
                    "AUC all-trained": f"{roc_auc_score(yy, all_trained[t]):.3f}",
                    "Fold AUC all-trained": fold_auc(yy, all_trained[t], folds60)[0],
                    "rho with age, matched-trained": f"{spearmanr(age60[non_dem], s[non_dem])[0]:+.2f}",
                    "rho with age, all-trained": f"{spearmanr(age60[non_dem], all_trained[t][non_dem])[0]:+.2f}",
                    "AAUC matched-trained": f"{ev.age_adjusted_auc(yy, s, age60):.3f}",
                    "AAUC all-trained": f"{ev.age_adjusted_auc(yy, all_trained[t], age60):.3f}"}
    results = {"n": int(old.sum()), "events": int(yy.sum()), "seeds": seeds,
               "source": str(args.runs_dir) if args.runs_dir is not None else AGEMATCHED_FILE,
               "test folds": [int((folds60 == k).sum()) for k in np.unique(folds60)],
               "age": {"demented": f"{age60[yy == 1].mean():.1f} ± {age60[yy == 1].std(ddof=1):.1f}",
                       "non-demented": f"{age60[yy == 0].mean():.1f} ± {age60[yy == 0].std(ddof=1):.1f}"},
               "references": {"Age alone": f"{roc_auc_score(yy, age60):.3f}", "LR-2": f"{roc_auc_score(yy, scores['LR-2'][old]):.3f}",
                              "MMSE alone": f"{roc_auc_score(yy, scores['MMSE alone'][old]):.3f}"},
               "networks": table}
    results["M1: matched-trained vs all-trained"] = probe.finish(
        {t: probe.compare(yy, matched[t], all_trained[t], nb, sd) for t in matched})
    results["M2: matched-trained vs age alone"] = family(yy, matched, age60, nb, sd)
    results["M3: matched-trained vs LR-2"] = family(yy, matched, scores["LR-2"][old], nb, sd)
    results["exploratory: matched-trained vs MMSE alone"] = family(yy, matched, scores["MMSE alone"][old], nb, sd)

    if args.features is not None:
        feats = pd.read_csv(args.features)
        feats["subject_id"] = feats["subject_id"].astype(str)
        feats = pd.DataFrame({"subject_id": ids.astype(str).to_numpy()}).merge(feats, on="subject_id", how="left", validate="1:1")
        x = feats.drop(columns="subject_id").to_numpy(dtype=float)
        if np.isnan(x).any():
            raise SystemExit("probe features missing for some matched subjects")
        oof, chosen = probe.probe_oof(x, yy, folds60, None, sd, probe.C_GRID)
        all_probe = scores[ev.PROBE][old]
        r = probe.compare(yy, oof, all_probe, nb, sd)
        r["p"] = ev.fmt_p(r.pop("p_raw"))
        fold_refit, per_refit = fold_auc(yy, oof, folds60)
        fold_all, per_all = fold_auc(yy, all_probe, folds60)
        results["probe refitted on the matched subjects"] = {
            "AUC": f"{roc_auc_score(yy, oof):.3f}", "95% CI": ev.bootstrap_ci(yy, oof, nb, sd), "C per fold": chosen,
            "Fold AUC": fold_refit, "per-fold AUCs": per_refit,
            "AUC all-fitted probe": f"{roc_auc_score(yy, all_probe):.3f}", "Fold AUC all-fitted probe": fold_all,
            "per-fold AUCs all-fitted probe": per_all, "vs all-fitted probe (pooled scores)": r,
            "rho with age, refitted": f"{spearmanr(age60[non_dem], oof[non_dem])[0]:+.2f}",
            "rho with age, all-fitted": f"{spearmanr(age60[non_dem], all_probe[non_dem])[0]:+.2f}",
            "AAUC refitted": f"{ev.age_adjusted_auc(yy, oof, age60):.3f}", "AAUC all-fitted": f"{ev.age_adjusted_auc(yy, all_probe, age60):.3f}"}

    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_dict(table, orient="index").to_csv(args.out / "table_s12_agematched.csv", index_label="Model")
    (args.out / "agematched.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(pd.DataFrame.from_dict(table, orient="index").to_string())
    print(json.dumps({k: v for k, v in results.items() if k not in ("networks",)}, indent=1))


if __name__ == "__main__":
    main()
