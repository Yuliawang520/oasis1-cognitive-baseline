"""Reproduce Table I, Table II, Fig. 1 and Fig. 2 of the paper.

Inputs
  --cohort  A CSV built from OASIS-1 by the notebooks (oasis_master_features_labels.csv),
            with columns subject_id, age, mmse, label, aseg_hippo_total, wbv_final.
            OASIS data cannot be redistributed, so this file is not part of the repository.
  --data    Directory with folds.csv and oof_predictions.csv (default: ../data).

The five interpretable baselines are refitted on the shared 5-fold split; the ANFIS and deep
models are read from their out-of-fold predictions. Every reproduced value is compared with
the value printed in the paper.

Usage: python scripts/evaluate.py --cohort path/to/oasis_master_features_labels.csv
"""
import argparse
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

REPO = Path(__file__).resolve().parents[1]
EVENTS = 93

MODELS = ["LR-3", "MMSE alone", "LR-1", "CART-3", "ANFIS-on-GT", "ANFIS-on-extracted", "LR-2",
          "DeiT-Small", "VGG-16", "ResNet-50", "DenseNet-121", "Swin-T"]
DEEP_COLUMNS = {"DeiT-Small": "deit_small", "VGG-16": "vgg16", "ResNet-50": "resnet50",
                "DenseNet-121": "densenet121", "Swin-T": "swin_t"}
PARAMS = {"LR-3": "4", "MMSE alone": "0", "LR-1": "2", "CART-3": "7", "ANFIS-on-GT": "126",
          "ANFIS-on-extracted": "126", "LR-2": "3", "DeiT-Small": "22M", "VGG-16": "138M",
          "ResNet-50": "25M", "DenseNet-121": "8M", "Swin-T": "28M"}
REFERENCES = {"LR-3": "LR-3", "MMSE": "MMSE alone", "ANFIS": "ANFIS-on-GT"}

# Table I as printed: AUC, 95% CI, Acc., Sens., Spec., Brier, p vs. LR-3 / MMSE / ANFIS.
PAPER = {
    "LR-3": ("0.9073", "[0.862, 0.946]", "0.854", "0.753", "0.925", "0.112", "---", "0.175", "0.0005"),
    "MMSE alone": ("0.8902", "[0.843, 0.931]", "n/a", "n/a", "n/a", "n/a", "0.175", "---", "0.011"),
    "LR-1": ("0.8827", "[0.831, 0.927]", "0.836", "0.720", "0.917", "0.124", "0.085", "0.176", "0.032"),
    "CART-3": ("0.8692", "[0.816, 0.917]", "0.805", "0.763", "0.835", "0.144", "0.093", "0.384", "0.131"),
    "ANFIS-on-GT": ("0.8232", "[0.759, 0.879]", "0.779", "0.634", "0.880", "0.193", "0.0005", "0.011", "---"),
    "ANFIS-on-extracted": ("0.7970", "[0.733, 0.855]", "0.730", "0.602", "0.820", "0.197", "<0.0001", "0.0003", "0.110"),
    "LR-2": ("0.7705", "[0.702, 0.831]", "0.735", "0.602", "0.827", "0.188", "<0.0001", "0.0008", "0.116"),
    "DeiT-Small": ("0.7648", "[0.701, 0.824]", "0.704", "0.634", "0.752", "0.193", "<0.0001", "0.0005", "0.144"),
    "VGG-16": ("0.7371", "[0.671, 0.799]", "0.681", "0.645", "0.707", "0.208", "<0.0001", "<0.0001", "0.040"),
    "ResNet-50": ("0.7257", "[0.660, 0.788]", "0.633", "0.247", "0.902", "0.233", "<0.0001", "<0.0001", "0.023"),
    "DenseNet-121": ("0.7168", "[0.651, 0.779]", "0.650", "0.462", "0.782", "0.216", "<0.0001", "<0.0001", "0.012"),
    "Swin-T": ("0.6773", "[0.609, 0.747]", "0.642", "0.355", "0.842", "0.226", "<0.0001", "<0.0001", "0.0014"),
}
COLUMNS = ["AUC", "95% CI", "Acc.", "Sens.", "Spec.", "Brier", "p vs. LR-3", "p vs. MMSE", "p vs. ANFIS"]

DARK_GREEN, MID_GREEN, LIGHT_GREEN = "#0b6e4f", "#3d9970", "#67c39b"
RED, ORANGE, BLUE, GREY = "#c1272d", "#e07a5f", "#2f6fb2", "#9aa5b1"


def anfis_probability(severity):
    """The ANFIS regresses CDR; map its output to a probability centred on the CDR 0 / 0.5 midpoint."""
    return 1.0 / (1.0 + np.exp(-(severity - 0.25) / 0.5))


def residualise_on_age(train, test, col):
    reg = LinearRegression().fit(train[["age"]], train[col])
    return train[col] - reg.predict(train[["age"]]), test[col] - reg.predict(test[["age"]])


def out_of_fold(cohort, features, make_model):
    """Train-fold-only age residualisation of the volumes and train-fold-only standardisation."""
    pred = np.zeros(len(cohort))
    for fold in sorted(cohort["fold"].unique()):
        train, test = cohort[cohort["fold"] != fold], cohort[cohort["fold"] == fold]
        x_train, x_test = pd.DataFrame(index=train.index), pd.DataFrame(index=test.index)
        for name, col in features.items():
            if name == "mmse":
                x_train[name], x_test[name] = train[col], test[col]
            else:
                x_train[name], x_test[name] = residualise_on_age(train, test, col)
        scaler = StandardScaler().fit(x_train)
        model = make_model().fit(scaler.transform(x_train), train["y"])
        pred[test.index] = model.predict_proba(scaler.transform(x_test))[:, 1]
    return pred


def load(cohort_csv, data_dir):
    cohort = pd.read_csv(cohort_csv)
    scores = pd.read_csv(data_dir / "oof_predictions.csv")
    cohort = scores[["subject_id", "fold"]].merge(cohort, on="subject_id", how="left", validate="1:1")
    needed = ["age", "mmse", "label", "aseg_hippo_total", "wbv_final"]
    missing = cohort[needed].isna().any(axis=1)
    if missing.any():
        raise SystemExit(f"{missing.sum()} subjects in folds.csv lack {needed} in {cohort_csv}")
    cohort["y"] = (cohort["label"] == "Demented").astype(int)

    three = {"hippo": "aseg_hippo_total", "wbv": "wbv_final", "mmse": "mmse"}
    logistic = lambda: LogisticRegression(max_iter=1000)
    tree = lambda: DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=0)
    scores_ = {  # rankings used for AUC
        "LR-3": out_of_fold(cohort, three, logistic),
        "MMSE alone": ((30 - cohort["mmse"]) / 30).to_numpy(),
        "LR-1": out_of_fold(cohort, {"mmse": "mmse"}, logistic),
        "CART-3": out_of_fold(cohort, three, tree),
        "ANFIS-on-GT": scores["anfis_gt_severity"].to_numpy(),
        "ANFIS-on-extracted": scores["anfis_extracted_severity"].to_numpy(),
        "LR-2": out_of_fold(cohort, {"hippo": "aseg_hippo_total", "wbv": "wbv_final"}, logistic),
    }
    for name, col in DEEP_COLUMNS.items():
        scores_[name] = scores[col].to_numpy()
    probs = dict(scores_)  # probabilities used at the 0.5 operating point and for the Brier score
    probs["ANFIS-on-GT"] = anfis_probability(scores_["ANFIS-on-GT"])
    probs["ANFIS-on-extracted"] = anfis_probability(scores_["ANFIS-on-extracted"])
    del probs["MMSE alone"]
    return cohort["y"].to_numpy(), scores_, probs


def delong_p(y, a, b):
    """Two-sided DeLong test for two correlated AUCs (fast algorithm of Sun and Xu, 2014)."""
    pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()
    v10, v01, aucs = [], [], []
    for s in (a, b):
        tz = rankdata(np.r_[s[pos], s[neg]])
        tx, ty = rankdata(s[pos]), rankdata(s[neg])
        aucs.append((tz[:m].sum() - m * (m + 1) / 2) / (m * n))
        v10.append((tz[:m] - tx) / n)
        v01.append(1 - (tz[m:] - ty) / m)
    cov = np.cov(v10) / m + np.cov(v01) / n
    se = np.sqrt(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    return 2 * norm.sf(abs(aucs[0] - aucs[1]) / se)


def bootstrap_ci(y, s, n_boot, seed):
    rng = np.random.default_rng(seed)
    aucs = []
    while len(aucs) < n_boot:
        idx = rng.integers(0, len(y), len(y))
        if 0 < y[idx].sum() < len(idx):
            aucs.append(roc_auc_score(y[idx], s[idx]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return f"[{lo:.3f}, {hi:.3f}]"


def fmt_p(p):
    return "<0.0001" if p < 1e-4 else f"{p:.4f}" if p < 0.01 else f"{p:.3f}"


def table_one(y, scores, probs, n_boot, seed):
    rows = {}
    for name in MODELS:
        s = scores[name]
        row = [f"{roc_auc_score(y, s):.4f}", bootstrap_ci(y, s, n_boot, seed)]
        if name in probs:
            p = probs[name]
            pred = (p >= 0.5).astype(int)
            tp, tn = ((pred == 1) & (y == 1)).sum(), ((pred == 0) & (y == 0)).sum()
            row += [f"{(tp + tn) / len(y):.3f}", f"{tp / y.sum():.3f}", f"{tn / (len(y) - y.sum()):.3f}",
                    f"{brier_score_loss(y, p):.3f}"]
        else:
            row += ["n/a"] * 4
        row += ["---" if name == ref else fmt_p(delong_p(y, s, scores[ref])) for ref in REFERENCES.values()]
        rows[name] = row
    return pd.DataFrame.from_dict(rows, orient="index", columns=COLUMNS)


def compare_with_paper(table):
    exact = [c for c in COLUMNS if c != "95% CI"]
    mismatches = [(m, c, table.loc[m, c], PAPER[m][COLUMNS.index(c)])
                  for m in MODELS for c in exact if table.loc[m, c] != PAPER[m][COLUMNS.index(c)]]
    worst_ci = max(abs(float(a) - float(b))
                   for m in MODELS
                   for a, b in zip(table.loc[m, "95% CI"].strip("[]").split(","),
                                   PAPER[m][1].strip("[]").split(",")))
    print(f"\nDeterministic columns: {len(MODELS) * len(exact) - len(mismatches)}/{len(MODELS) * len(exact)} "
          f"values identical to the paper.")
    for m, c, got, want in mismatches:
        print(f"   {m} / {c}: reproduced {got}, paper {want}")
    print(f"Bootstrap 95% CIs: largest difference from the paper is {worst_ci:.3f} "
          "(resampling noise; the paper's random seed is not recorded).")
    return not mismatches


def auc3(name):
    """Round the Table I value to 3 d.p. the way the text does (e.g. 0.7705 -> 0.771)."""
    return str(Decimal(PAPER[name][0]).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def style():
    plt.rcParams.update({
        "font.family": ["Arial", "Liberation Sans", "DejaVu Sans"], "font.size": 7, "axes.labelsize": 8, "legend.fontsize": 7,
        "legend.title_fontsize": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "pdf.fonttype": 42, "lines.solid_capstyle": "butt",
    })


def plot_roc(y, scores, path):
    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    ax.plot([0, 1], [0, 1], color="0.45", lw=0.6, ls=":", zorder=1)
    deep_aucs = [auc3(name) for name in DEEP_COLUMNS]
    for i, name in enumerate(DEEP_COLUMNS):
        fpr, tpr, _ = roc_curve(y, scores[name])
        ax.plot(fpr, tpr, color=GREY, lw=0.6, alpha=0.85, zorder=2,
                label=f"Deep baselines ×5 ({min(deep_aucs)}–{max(deep_aucs)})" if i == 0 else None)
    curves = [  # drawn bottom to top; the legend lists them by AUC
        ("LR-2", "LR-2: imaging only", BLUE, 1.1, "-"),
        ("ANFIS-on-extracted", "ANFIS-on-extracted", ORANGE, 1.0, "--"),
        ("ANFIS-on-GT", "ANFIS-on-GT", RED, 1.3, "-"),
        ("CART-3", "CART-3: depth 3", LIGHT_GREEN, 0.9, "--"),
        ("LR-1", "LR-1: MMSE only", MID_GREEN, 1.0, "--"),
        ("LR-3", "LR-3: hippo + WBV + MMSE", DARK_GREEN, 1.3, "-"),
        ("MMSE alone", "MMSE alone, no model", "black", 1.5, "-"),
    ]
    for name, label, color, lw, ls in curves:
        fpr, tpr, _ = roc_curve(y, scores[name])
        ax.plot(fpr, tpr, color=color, lw=lw, ls=ls, zorder=3, label=f"{label} ({auc3(name)})")
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: -float(labels[i].rsplit("(", 1)[1].split("–")[-1].rstrip(")")))
    ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower right",
              title="Model (AUC)", frameon=True, framealpha=0.95, edgecolor="0.75", fancybox=False,
              borderpad=0.4, labelspacing=0.28, handlelength=2.2, handletextpad=0.5, borderaxespad=0.4)
    ax.set(xlim=(-0.01, 1.01), ylim=(-0.01, 1.01), xlabel="False positive rate", ylabel="True positive rate")
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
    fig.tight_layout(pad=0.3)
    fig.savefig(path)
    plt.close(fig)


def plot_calibration(y, probs, path):
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ax.plot([0, 1], [0, 1], color="black", lw=0.7, ls=":", label="Perfect calibration", zorder=1)
    for name, color in [("LR-3", DARK_GREEN), ("CART-3", LIGHT_GREEN), ("ANFIS-on-GT", RED),
                        ("LR-2", BLUE), ("ResNet-50", GREY)]:
        observed, predicted = calibration_curve(y, probs[name], n_bins=6, strategy="quantile")
        ax.plot(predicted, observed, color=color, lw=1.1, marker="o", ms=3, zorder=2,
                label=f"{name} ({brier_score_loss(y, probs[name]):.3f})")
    ax.legend(loc="upper left", title="Model (Brier score)", frameon=True, framealpha=0.95,
              edgecolor="0.75", fancybox=False, borderpad=0.4, labelspacing=0.28, handlelength=2.0,
              handletextpad=0.5, borderaxespad=0.4)
    ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02), xlabel="Mean predicted probability",
           ylabel="Observed fraction demented")
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
    fig.tight_layout(pad=0.3)
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=REPO / "data")
    parser.add_argument("--out", type=Path, default=REPO / "results")
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    y, scores, probs = load(args.cohort, args.data)
    assert y.sum() == EVENTS and len(y) == 226, "unexpected cohort"
    table = table_one(y, scores, probs, args.n_boot, args.seed)
    table.insert(0, "Params", [PARAMS[m] for m in table.index])
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("Table I (reproduced)\n", table.to_string())
    table.to_csv(args.out / "table1_reproduced.csv", index_label="Model")

    epv = pd.DataFrame({"Params": [4, 7, 126]}, index=["LR-3", "CART-3", "ANFIS"])
    epv["EPV"] = (EVENTS / epv["Params"]).round(2)
    print("\nTable II (EPV, E = 93)\n", epv.to_string())

    style()
    plot_roc(y, scores, args.out / "fig_roc.pdf")
    plot_calibration(y, probs, args.out / "fig_calib.pdf")
    print(f"\nFigures written to {args.out}")
    if not compare_with_paper(table.drop(columns="Params")):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
