"""Robustness analyses of the paper: ANFIS ablations, repeated cross-validation and decision curves.

Inputs are those of evaluate.py. The ANFIS (scripts/anfis.py) reproduces the released out-of-fold
outputs of the notebook, so the ablations differ from the paper's ANFIS in one setting at a time.
Requires torch and imbalanced-learn in addition to the packages used by evaluate.py.

Usage: python scripts/robustness.py --cohort path/to/oasis_master_features_labels.csv [--repeats 100]
"""
import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anfis import Config, fit_predict, n_parameters  # noqa: E402
from evaluate import (BLACK, BLUE, DASH_DOT, GREEN, GREY, REPO, SKY, VERMILION,  # noqa: E402
                      bootstrap_delta_ci, delong_p, fmt_p, out_of_fold, recalibrate, style)

THREE = ("hippo", "wbv", "mmse")
ABLATIONS = [  # (name, inputs, configuration, reference model)
    ("Original (paper)", THREE, Config(), "LR-3"),
    ("Without SMOTE", THREE, Config(smote=False), "LR-3"),
    ("Binary target", THREE, Config(target="binary"), "LR-3"),
    ("Binary target without SMOTE", THREE, Config(target="binary", smote=False), "LR-3"),
    ("8 rules", THREE, Config(n_mfs=2), "LR-3"),
    ("64 rules", THREE, Config(n_mfs=4), "LR-3"),
    ("Minimal", THREE, Config(n_mfs=2, target="binary", smote=False, class_weights=False), "LR-3"),
    ("Imaging only", ("hippo", "wbv"), Config(), "LR-2"),
]
MINIMAL = next(cfg for name, _, cfg, _ in ABLATIONS if name == "Minimal")
SMOTE_SEEDS = range(10)
THRESHOLDS = np.round(np.arange(0.05, 0.801, 0.01), 2)
DCA_POINTS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)


def load(cohort_csv, data_dir):
    """The 226 subjects in the row order of the ANFIS notebook (which SMOTE needs for an exact
    reproduction), with the shared fold assignment and the released ANFIS-on-GT output."""
    master = pd.read_csv(cohort_csv)
    cohort = master.dropna(subset=["aseg_hippo_total", "wbv_final", "mmse", "age", "cdr_numeric"]).reset_index(drop=True)
    cohort = cohort.merge(pd.read_csv(data_dir / "folds.csv"), on="subject_id", how="inner", validate="1:1")
    released = pd.read_csv(data_dir / "oof_predictions.csv")
    released["eval_pos"] = np.arange(len(released))  # row order of evaluate.py, used for every statistic
    cohort = cohort.merge(released[["subject_id", "eval_pos", "anfis_gt_severity"]], on="subject_id", validate="1:1")
    cohort["y"] = (cohort["label"] == "Demented").astype(int)
    assert len(cohort) == 226 and cohort["y"].sum() == 93, "unexpected cohort"
    return cohort


def anfis_oof(cohort, folds, inputs, cfg):
    """Train-fold-only age residualisation of the volumes and standardisation, as in the notebook."""
    cols = {"hippo": "aseg_hippo_total", "wbv": "wbv_final", "mmse": "mmse"}
    x_raw = cohort[[cols[c] for c in inputs]].to_numpy(dtype=np.float32)
    age = cohort["age"].to_numpy(dtype=np.float32)
    cdr = cohort["cdr_numeric"].to_numpy(dtype=np.float32)
    label = cohort["y"].to_numpy()
    out = np.zeros(len(cohort), dtype=np.float32)
    for k in np.unique(folds):
        train, test = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        x = x_raw.copy()
        for j, name in enumerate(inputs):
            if name != "mmse":
                reg = LinearRegression().fit(age[train].reshape(-1, 1), x_raw[train, j])
                x[:, j] = x_raw[:, j] - reg.predict(age.reshape(-1, 1))
        x = StandardScaler().fit(x[train]).transform(x)
        out[test] = fit_predict(x[train], cdr[train], label[train], x[test], cfg)
    return out.astype(float)


def tabular_oof(cohort, folds):
    c = cohort.assign(fold=folds)
    three = {"hippo": "aseg_hippo_total", "wbv": "wbv_final", "mmse": "mmse"}
    logistic = lambda: LogisticRegression(max_iter=1000)
    tree = lambda: DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=0)
    return {"MMSE alone": ((30 - cohort["mmse"]) / 30).to_numpy(),
            "LR-1": out_of_fold(c, {"mmse": "mmse"}, logistic),
            "LR-2": out_of_fold(c, {"hippo": "aseg_hippo_total", "wbv": "wbv_final"}, logistic),
            "LR-3": out_of_fold(c, three, logistic),
            "CART-3": out_of_fold(c, three, tree)}


# ---------------------------------------------------------------- ablations

def ablation_table(cohort, y, folds, tab, n_boot, seed):
    """Statistics use the row order of evaluate.py, and the original configuration uses the released
    outputs, so that its row matches the paper's primary comparison F3 exactly."""
    order = np.argsort(cohort["eval_pos"].to_numpy())
    rows, scores = {}, {}
    for name, inputs, cfg, ref in ABLATIONS:
        s = anfis_oof(cohort, folds, inputs, cfg)
        scores[name] = s
        if name == "Original (paper)":
            s = cohort["anfis_gt_severity"].to_numpy()
        yy, a, b = y[order], tab[ref][order], s[order]
        auc = roc_auc_score(yy, b)
        seed_aucs = [roc_auc_score(y, anfis_oof(cohort, folds, inputs, replace(cfg, smote_seed=k)))
                     for k in SMOTE_SEEDS] if cfg.smote else [auc]
        rows[name] = {"Rules": cfg.n_mfs ** len(inputs), "Params": n_parameters(len(inputs), cfg.n_mfs),
                      "AUC": f"{auc:.3f}", "SMOTE-seed range": f"{min(seed_aucs):.3f}-{max(seed_aucs):.3f}",
                      "Reference": ref, "dAUC (ref - ANFIS)": f"{roc_auc_score(yy, a) - auc:+.3f}",
                      "95% CI": bootstrap_delta_ci(yy, a, b, n_boot, seed), "p": fmt_p(delong_p(yy, a, b))}
        print(f"   {name}: AUC {auc:.4f} (seeds {min(seed_aucs):.3f}-{max(seed_aucs):.3f}); vs {ref} p {rows[name]['p']}")
    return pd.DataFrame.from_dict(rows, orient="index"), scores


# ---------------------------------------------------------------- repeated cross-validation

COMPARISONS = [  # (label, a, b): difference a - b
    ("LR-3 vs MMSE alone", "LR-3", "MMSE alone"),
    ("CART-3 vs MMSE alone", "CART-3", "MMSE alone"),
    ("ANFIS-on-GT vs MMSE alone", "ANFIS-on-GT", "MMSE alone"),
    ("LR-2 vs MMSE alone", "LR-2", "MMSE alone"),
    ("LR-3 vs LR-1", "LR-3", "LR-1"),
    ("LR-3 vs ANFIS-on-GT", "LR-3", "ANFIS-on-GT"),
    ("LR-3 vs minimal ANFIS", "LR-3", "Minimal ANFIS"),
]


def repeated_cv(cohort, y, repeats, shared):
    aucs, deltas, pvals = [], [], []
    start = time.time()
    for r in range(repeats):
        folds = np.zeros(len(y), int)
        for k, (_, test) in enumerate(StratifiedKFold(5, shuffle=True, random_state=r).split(np.zeros(len(y)), y)):
            folds[test] = k
        s = tabular_oof(cohort, folds)
        s["ANFIS-on-GT"] = anfis_oof(cohort, folds, THREE, Config())
        s["Minimal ANFIS"] = anfis_oof(cohort, folds, THREE, MINIMAL)
        aucs.append({m: roc_auc_score(y, v) for m, v in s.items()})
        deltas.append({c: aucs[-1][a] - aucs[-1][b] for c, a, b in COMPARISONS})
        pvals.append({c: delong_p(y, s[a], s[b]) for c, a, b in COMPARISONS})
        if (r + 1) % 10 == 0:
            print(f"   repeat {r + 1}/{repeats} ({time.time() - start:.0f} s)")
    aucs, deltas, pvals = pd.DataFrame(aucs), pd.DataFrame(deltas), pd.DataFrame(pvals)

    def q(v):
        lo, med, hi = np.percentile(v, [2.5, 50, 97.5])
        return med, lo, hi

    model_rows = {m: dict(zip(("Median AUC", "2.5%", "97.5%"), (f"{x:.3f}" for x in q(aucs[m])))) for m in aucs}
    comp_rows = {}
    for c, a, b in COMPARISONS:
        med, lo, hi = q(deltas[c])
        sign = np.sign(shared[a] - shared[b])
        comp_rows[c] = {"Shared split": f"{shared[a] - shared[b]:+.3f}", "Median": f"{med:+.3f}",
                        "2.5%": f"{lo:+.3f}", "97.5%": f"{hi:+.3f}",
                        "Same sign (%)": f"{100 * np.mean(np.sign(deltas[c]) == sign):.0f}",
                        "p < 0.05 (%)": f"{100 * np.mean(pvals[c] < 0.05):.0f}"}
    return pd.DataFrame.from_dict(model_rows, orient="index"), pd.DataFrame.from_dict(comp_rows, orient="index")


# ---------------------------------------------------------------- decision curves

def net_benefit(y, p, pt):
    flagged = p >= pt
    tp, fp = np.sum(flagged & (y == 1)), np.sum(flagged & (y == 0))
    return (tp - fp * pt / (1 - pt)) / len(y)


def decision_curves(y, probs, n_boot, seed):
    curves = {m: np.array([net_benefit(y, p, t) for t in THRESHOLDS]) for m, p in probs.items()}
    prevalence = y.mean()
    curves["Treat all"] = prevalence - (1 - prevalence) * THRESHOLDS / (1 - THRESHOLDS)
    rng = np.random.default_rng(seed)
    boot = {t: [] for t in DCA_POINTS}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        for t in DCA_POINTS:
            boot[t].append(net_benefit(y[idx], probs["LR-3"][idx], t) - net_benefit(y[idx], probs["LR-1"][idx], t))
    rows = {}
    for t in DCA_POINTS:
        lo, hi = np.percentile(boot[t], [2.5, 97.5])
        rows[f"{t:.1f}"] = {m: f"{net_benefit(y, probs[m], t):.3f}" for m in probs}
        rows[f"{t:.1f}"]["Treat all"] = f"{prevalence - (1 - prevalence) * t / (1 - t):.3f}"
        rows[f"{t:.1f}"]["LR-3 - LR-1"] = f"{net_benefit(y, probs['LR-3'], t) - net_benefit(y, probs['LR-1'], t):+.3f}"
        rows[f"{t:.1f}"]["95% CI"] = f"[{lo:+.3f}, {hi:+.3f}]"
    return curves, pd.DataFrame.from_dict(rows, orient="index")


def plot_dca(curves, path):
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    lines = [("LR-3", "LR-3, MMSE + volumes", GREEN, 1.3, "-"), ("LR-1", "LR-1, MMSE only", SKY, 1.2, (0, (5, 2))),
             ("ANFIS-on-GT", "ANFIS-on-GT", VERMILION, 1.2, DASH_DOT), ("LR-2", "LR-2, volumes only", BLUE, 1.1, "-"),
             ("Treat all", "Treat all", GREY, 0.9, (0, (1.2, 1.2)))]
    for name, label, color, lw, ls in reversed(lines):  # LR-3 drawn on top
        ax.plot(THRESHOLDS, curves[name], color=color, lw=lw, ls=ls, label=label)
    ax.axhline(0, color=BLACK, lw=0.6, label="Treat none")
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(label) for _, label, *_ in lines] + [labels.index("Treat none")]
    ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper right", frameon=True,
              framealpha=0.95, edgecolor="0.75", fancybox=False, borderpad=0.4, labelspacing=0.25,
              handlelength=2.4, handletextpad=0.5, borderaxespad=0.3)
    ax.set(xlim=(0.05, 0.8), ylim=(-0.05, 0.45), xlabel="Threshold probability", ylabel="Net benefit")
    fig.tight_layout(pad=0.3)
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=REPO / "data")
    parser.add_argument("--out", type=Path, default=REPO / "results")
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cohort = load(args.cohort, args.data)
    y, folds = cohort["y"].to_numpy(), cohort["fold"].to_numpy()
    tab = tabular_oof(cohort, folds)
    released_gt = cohort["anfis_gt_severity"].to_numpy()
    rerun = anfis_oof(cohort, folds, THREE, Config())
    print(f"ANFIS-on-GT re-run vs released outputs: max difference {np.max(np.abs(rerun - released_gt)):.1e}, "
          f"AUC {roc_auc_score(y, rerun):.4f} vs {roc_auc_score(y, released_gt):.4f}")
    order = np.argsort(cohort["eval_pos"].to_numpy())  # row order of evaluate.py

    results = {}
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print("\nANFIS ablations on the shared split")
        ablation, ablation_scores = ablation_table(cohort, y, folds, tab, args.n_boot, args.seed)
        print(ablation.to_string())
        ablation.to_csv(args.out / "table_ablation.csv", index_label="Configuration")
        results["ablation"] = ablation.to_dict(orient="index")

        probs = {"LR-1": tab["LR-1"], "LR-2": tab["LR-2"], "LR-3": tab["LR-3"],
                 "ANFIS-on-GT": recalibrate(released_gt, y, folds)}
        curves, dca = decision_curves(y[order], {m: p[order] for m, p in probs.items()}, args.n_boot, args.seed)
        print(f"\nNet benefit (prevalence {y.mean():.3f})\n", dca.to_string())
        dca.to_csv(args.out / "table_dca.csv", index_label="Threshold")
        results["dca"] = dca.to_dict(orient="index")

        print(f"\nRepeated cross-validation ({args.repeats} stratified five-fold splits)")
        shared = {m: roc_auc_score(y, v) for m, v in tab.items()}
        shared.update({"ANFIS-on-GT": roc_auc_score(y, released_gt),
                       "Minimal ANFIS": roc_auc_score(y, ablation_scores["Minimal"])})
        models, comparisons = repeated_cv(cohort, y, args.repeats, shared)
        print(models.to_string(), "\n", comparisons.to_string())
        models.to_csv(args.out / "table_repeated_cv_models.csv", index_label="Model")
        comparisons.to_csv(args.out / "table_repeated_cv_comparisons.csv", index_label="Comparison")
        results["repeated_cv"] = {"repeats": args.repeats, "models": models.to_dict(orient="index"),
                                  "comparisons": comparisons.to_dict(orient="index")}

    style()
    plot_dca(curves, args.out / "fig_dca.pdf")
    (args.out / "robustness.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nTables, fig_dca.pdf and robustness.json written to {args.out}")


if __name__ == "__main__":
    main()
