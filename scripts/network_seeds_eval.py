"""Seed sensitivity of the five slice networks on OASIS-1 (Q3 of docs/prespecification_networks_adni.md).

Reads the seeded re-runs written by notebooks/slice_networks_multiseed_colab.ipynb (COHORT = 'oasis'):
  runs_oasis/ensemble_oasis_<model>_poolfull.csv   columns subject_id, label, fold, seed0.., prob_demented
and the original benchmark (data/oof_predictions.csv, the cohort file) through evaluate.load, then reports

  * per network: original AUC, per-seed AUCs (mean ± SD, range), whether the original run lies inside the seed
    range, and the seed-ensemble AUC with its CI;
  * families F1 (MMSE alone vs the ten models with imaging inputs, Holm over ten) and F4 (age alone vs the five
    networks, all subjects and aged 60 or over, Holm over five) re-evaluated with the seed ensembles in place of the
    original network runs, and once more for every single seed;
  * a stability flag per network and family: "stable" when direction and Holm significance (0.05) agree between the
    original run, the ensemble and every seed.

Decision rule of the plan: an original run stays the primary result if it lies within the seed range.

Usage: python scripts/network_seeds_eval.py --cohort path/to/oasis_master_features_labels.csv --runs-dir path/to/runs_oasis
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as ev  # noqa: E402

RUN_NAMES = {"DeiT-Small": "deit_small", "VGG-16": "vgg16", "ResNet-50": "resnet50",
             "DenseNet-121": "densenet121", "Swin-T": "swin_t"}


def load_runs(runs_dir: Path, cohort: pd.DataFrame) -> tuple[dict, dict, list[str]]:
    ensembles, per_seed, seeds = {}, {}, []
    for title, name in RUN_NAMES.items():
        f = runs_dir / f"ensemble_oasis_{name}_poolfull.csv"
        if not f.exists():
            print(f"missing {f.name}")
            continue
        df = pd.read_csv(f)
        df["subject_id"] = df["subject_id"].astype(str)
        merged = cohort[["subject_id"]].merge(df, on="subject_id", how="left", validate="1:1")
        if merged["prob_demented"].isna().any():
            raise SystemExit(f"{f.name}: {int(merged['prob_demented'].isna().sum())} subjects without a score")
        seed_cols = sorted(c for c in df.columns if c.startswith("seed"))
        seeds = seeds or seed_cols
        ensembles[title] = merged["prob_demented"].to_numpy()
        per_seed[title] = {c: merged[c].to_numpy() for c in seed_cols}
    return ensembles, per_seed, seeds


def family_f1(y, scores):
    """MMSE alone vs each model with imaging inputs: direction and Holm-adjusted p over the ten comparisons."""
    ps = [ev.delong_p(y, scores["MMSE alone"], scores[m]) for m in ev.IMAGING_INPUT]
    deltas = [roc_auc_score(y, scores[m]) - roc_auc_score(y, scores["MMSE alone"]) for m in ev.IMAGING_INPUT]
    return {m: {"dAUC": d, "Holm p": h} for m, d, h in zip(ev.IMAGING_INPUT, deltas, ev.holm(ps))}


def family_f4(y, age, scores):
    """Age alone vs each network, on all subjects and on those aged 60 or over (Holm over five per cohort)."""
    out = {}
    for label, mask in (("all", np.ones(len(y), bool)), ("60+", age >= 60)):
        yy, a = y[mask], age[mask]
        ps = [ev.delong_p(yy, a, scores[m][mask]) for m in RUN_NAMES]
        deltas = [roc_auc_score(yy, scores[m][mask]) - roc_auc_score(yy, a) for m in RUN_NAMES]
        out[label] = {m: {"dAUC": d, "Holm p": h} for m, d, h in zip(RUN_NAMES, deltas, ev.holm(ps))}
    return out


def verdict(entries: list[dict]) -> str:
    """'stable' if every entry has the same sign of dAUC and the same Holm significance at 0.05."""
    signs = {np.sign(e["dAUC"]) for e in entries}
    sig = {e["Holm p"] < 0.05 for e in entries}
    return "stable" if len(signs) == 1 and len(sig) == 1 else "unstable"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cohort", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=ev.REPO / "data")
    ap.add_argument("--runs-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ev.REPO / "results")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cohort, y, folds, scores, _ = ev.load(args.cohort, args.data)
    age = cohort["age"].to_numpy(dtype=float)
    ensembles, per_seed, seeds = load_runs(args.runs_dir, cohort)
    if not ensembles:
        raise SystemExit("no seeded runs found")

    rows = {}
    for title in RUN_NAMES:
        if title not in ensembles:
            continue
        orig = roc_auc_score(y, scores[title])
        seed_aucs = {c: roc_auc_score(y, v) for c, v in per_seed[title].items()}
        vals = np.array(list(seed_aucs.values()))
        rows[title] = {
            "Original AUC": f"{orig:.3f}",
            **{c: f"{a:.3f}" for c, a in seed_aucs.items()},
            "Seed mean ± SD": f"{vals.mean():.3f} ± {vals.std(ddof=1):.3f}" if len(vals) > 1 else f"{vals[0]:.3f}",
            "Seed range": f"{vals.min():.3f}-{vals.max():.3f}",
            "Original inside range": bool(vals.min() <= orig <= vals.max()),
            "Ensemble AUC": f"{roc_auc_score(y, ensembles[title]):.3f}",
            "Ensemble 95% CI": ev.bootstrap_ci(y, ensembles[title], args.n_boot, args.seed),
            "Ensemble 60+ AUC": f"{roc_auc_score(y[age >= 60], ensembles[title][age >= 60]):.3f}",
        }
    table = pd.DataFrame.from_dict(rows, orient="index")

    # families with the original runs, the ensembles, and each seed
    variants = {"original": dict(scores)}
    variants["ensemble"] = {**scores, **ensembles}
    for c in seeds:
        variants[c] = {**scores, **{t: per_seed[t][c] for t in ensembles}}
    f1 = {v: family_f1(y, s) for v, s in variants.items()}
    f4 = {v: family_f4(y, age, s) for v, s in variants.items()}

    stability = {}
    for title in ensembles:
        stability[title] = {
            "F1 (vs MMSE)": verdict([f1[v][title] for v in variants]),
            "F4 all (vs age)": verdict([f4[v]["all"][title] for v in variants]),
            "F4 60+ (vs age)": verdict([f4[v]["60+"][title] for v in variants]),
        }
    # MMSE-vs-every-model conclusion of the paper: no model significantly exceeds the MMSE
    none_exceed = {v: bool(all(not (e["dAUC"] > 0 and e["Holm p"] < 0.05) for e in f1[v].values())) for v in variants}

    def fmt(fam):
        return {v: {m: {"dAUC": f"{e['dAUC']:+.3f}", "Holm p": ev.fmt_p(e["Holm p"])} for m, e in d.items()} for v, d in fam.items()}

    results = {"seeds": seeds, "networks": rows, "stability": stability,
               "no model significantly exceeds the MMSE": none_exceed,
               "F1 by variant": fmt(f1),
               "F4 by variant": {v: {lab: {m: {"dAUC": f"{e['dAUC']:+.3f}", "Holm p": ev.fmt_p(e["Holm p"])} for m, e in d.items()}
                                     for lab, d in fam.items()} for v, fam in f4.items()}}
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "table_s8_seed_sensitivity.csv", index_label="Model")
    (args.out / "seed_sensitivity.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(table.to_string())
    print(json.dumps({k: results[k] for k in ("stability", "no model significantly exceeds the MMSE")}, indent=1))


if __name__ == "__main__":
    main()
