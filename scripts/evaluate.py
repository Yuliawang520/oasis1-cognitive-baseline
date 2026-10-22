"""Reproduce every table and figure of the paper.

Inputs
  --cohort        A CSV built from OASIS-1 by the notebooks (oasis_master_features_labels.csv),
                  with columns subject_id, age, mmse, label, aseg_hippo_total, wbv_final.
  --demographics  Optional: the OASIS-1 cross-sectional spreadsheet, for the cohort table.
  --data          Directory with folds.csv and oof_predictions.csv (default: ../data).
OASIS data cannot be redistributed, so neither input is part of the repository.

The five interpretable baselines are refitted on the shared 5-fold split; the ANFIS and deep
models are read from their out-of-fold predictions. Every reproduced value is compared with
the value printed in the paper.

Usage: python scripts/evaluate.py --cohort path/to/oasis_master_features_labels.csv
"""
import argparse
import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2, norm, rankdata
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

REPO = Path(__file__).resolve().parents[1]
EVENTS = 93

MODELS = ["LR-3", "MMSE alone", "LR-1", "CART-3", "ANFIS-on-GT", "ANFIS-on-extracted", "LR-2",
          "DeiT-Small", "VGG-16", "ResNet-50", "DenseNet-121", "Swin-T"]
GROUPS = {"Cognitive score only": ["MMSE alone", "LR-1"],
          "Cognitive score and imaging": ["LR-3", "CART-3", "ANFIS-on-GT", "ANFIS-on-extracted"],
          "Imaging only": ["LR-2", "DeiT-Small", "VGG-16", "ResNet-50", "DenseNet-121", "Swin-T"]}
IMAGING_INPUT = GROUPS["Cognitive score and imaging"] + GROUPS["Imaging only"]
DEEP_COLUMNS = {"DeiT-Small": "deit_small", "VGG-16": "vgg16", "ResNet-50": "resnet50",
                "DenseNet-121": "densenet121", "Swin-T": "swin_t"}
PARAMS = {"LR-3": "4", "MMSE alone": "0", "LR-1": "2", "CART-3": "7", "ANFIS-on-GT": "126",
          "ANFIS-on-extracted": "126", "LR-2": "3", "DeiT-Small": "22M", "VGG-16": "138M",
          "ResNet-50": "25M", "DenseNet-121": "8M", "Swin-T": "28M"}

# Okabe-Ito colours, distinguishable with colour-vision deficiency.
BLACK, ORANGE, SKY, GREEN, BLUE, VERMILION, PURPLE, GREY = (
    "#000000", "#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#999999")
DASH_DOT = (0, (5, 1.5, 1.2, 1.5))

# Values printed in the paper, filled in from a verified run (see check_against_paper).
PAPER = json.loads((REPO / "scripts" / "paper_values.json").read_text(encoding="utf-8")) \
    if (REPO / "scripts" / "paper_values.json").exists() else None


# ---------------------------------------------------------------- data and models

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


def recalibrate(severity, y, folds):
    """Map the CDR-scale ANFIS output to a probability: for each test fold, a logistic (Platt)
    map fitted on the out-of-fold outputs of the other four folds."""
    p = np.zeros(len(severity))
    for k in np.unique(folds):
        train = folds != k
        m = LogisticRegression(penalty=None).fit(severity[train, None], y[train])
        p[~train] = m.predict_proba(severity[~train, None])[:, 1]
    return p


def load(cohort_csv, data_dir):
    scores_csv = pd.read_csv(data_dir / "oof_predictions.csv")
    cohort = scores_csv[["subject_id", "fold"]].merge(pd.read_csv(cohort_csv), on="subject_id",
                                                     how="left", validate="1:1")
    needed = ["age", "mmse", "label", "aseg_hippo_total", "wbv_final"]
    missing = cohort[needed].isna().any(axis=1)
    if missing.any():
        raise SystemExit(f"{missing.sum()} subjects in folds.csv lack {needed} in {cohort_csv}")
    cohort["y"] = (cohort["label"] == "Demented").astype(int)
    y, folds = cohort["y"].to_numpy(), cohort["fold"].to_numpy()

    three = {"hippo": "aseg_hippo_total", "wbv": "wbv_final", "mmse": "mmse"}
    logistic = lambda: LogisticRegression(max_iter=1000)
    tree = lambda: DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=0)
    scores = {  # rankings used for AUC
        "LR-3": out_of_fold(cohort, three, logistic),
        "MMSE alone": ((30 - cohort["mmse"]) / 30).to_numpy(),
        "LR-1": out_of_fold(cohort, {"mmse": "mmse"}, logistic),
        "CART-3": out_of_fold(cohort, three, tree),
        "ANFIS-on-GT": scores_csv["anfis_gt_severity"].to_numpy(),
        "ANFIS-on-extracted": scores_csv["anfis_extracted_severity"].to_numpy(),
        "LR-2": out_of_fold(cohort, {"hippo": "aseg_hippo_total", "wbv": "wbv_final"}, logistic),
    }
    for name, col in DEEP_COLUMNS.items():
        scores[name] = scores_csv[col].to_numpy()
    probs = {k: v for k, v in scores.items() if k != "MMSE alone"}  # probabilities for operating point, Brier
    for name in ("ANFIS-on-GT", "ANFIS-on-extracted"):
        probs[name] = recalibrate(scores[name], y, folds)
    return cohort, y, folds, scores, probs


# ---------------------------------------------------------------- statistics

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


def holm(ps):
    ps = np.asarray(ps, float)
    adjusted, running = np.empty_like(ps), 0.0
    for rank, i in enumerate(np.argsort(ps)):
        running = max(running, min(1.0, ps[i] * (len(ps) - rank)))
        adjusted[i] = running
    return adjusted


def bootstrap_ci(y, s, n_boot, seed):
    rng = np.random.default_rng(seed)
    aucs = []
    while len(aucs) < n_boot:
        idx = rng.integers(0, len(y), len(y))
        if 0 < y[idx].sum() < len(idx):
            aucs.append(roc_auc_score(y[idx], s[idx]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return f"[{lo:.3f}, {hi:.3f}]"


def bootstrap_delta_ci(y, a, b, n_boot, seed):
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        deltas.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return f"[{lo:+.3f}, {hi:+.3f}]"


def likelihood_ratio(cohort, mask):
    """Full-sample test of the two imaging coefficients added to the MMSE, with odds ratios."""
    c = cohort[mask].reset_index(drop=True)
    y = c["y"].to_numpy()
    z = pd.DataFrame({"hippo": residualise_on_age(c, c, "aseg_hippo_total")[0],
                      "wbv": residualise_on_age(c, c, "wbv_final")[0], "mmse": c["mmse"]})
    z = (z - z.mean()) / z.std(ddof=0)

    def fit(cols):
        model = LogisticRegression(penalty=None, max_iter=5000).fit(z[cols], y)
        p = model.predict_proba(z[cols])[:, 1]
        return np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)), model, p

    ll_mmse, _, _ = fit(["mmse"])
    ll_full, model, p = fit(["hippo", "wbv", "mmse"])
    stat = 2 * (ll_full - ll_mmse)
    x = np.c_[np.ones(len(y)), z.to_numpy()]
    cov = np.linalg.inv(x.T @ (x * (p * (1 - p))[:, None]))
    beta = np.r_[model.intercept_, model.coef_[0]]
    odds = {}
    for j, name in enumerate(["hippo", "wbv", "mmse"], 1):  # per SD decrease
        se = np.sqrt(cov[j, j])
        odds[name] = f"{np.exp(-beta[j]):.2f} ({np.exp(-beta[j] - 1.96 * se):.2f}-{np.exp(-beta[j] + 1.96 * se):.2f})"
    phi = y.mean()
    ll_null = len(y) * (phi * np.log(phi) + (1 - phi) * np.log(1 - phi))
    r2cs = 1 - np.exp(2 * (ll_null - ll_full) / len(y))
    return {"chi2": f"{stat:.1f}", "p": fmt_p(chi2.sf(stat, 2)), "odds_ratios": odds,
            "r2cs": r2cs, "max_r2cs": 1 - np.exp(2 * ll_null / len(y))}


def riley_n(params, r2cs, max_r2cs, prevalence):
    """Riley et al. (2019) criteria (i) shrinkage >= 0.9 and (ii) optimism <= 0.05; and (iii) overall risk."""
    n1 = params / ((0.9 - 1) * np.log(1 - r2cs / 0.9))
    s = r2cs / (r2cs + 0.05 * max_r2cs)
    n2 = params / ((s - 1) * np.log(1 - r2cs / s))
    n3 = (1.96 / 0.05) ** 2 * prevalence * (1 - prevalence)
    return int(np.ceil(max(n1, n2))), int(np.ceil(n3))


def fmt_p(p):
    return "<0.0001" if p < 1e-4 else f"{p:.4f}" if p < 0.01 else f"{p:.3f}"


def fmt2(x):
    s = f"{x:.2f}"
    return "0.00" if s == "-0.00" else s


# ---------------------------------------------------------------- tables

def cohort_table(cohort, demographics_xlsx):
    oasis = pd.read_excel(demographics_xlsx)
    oasis["subject_id"] = oasis["ID"].astype(str).str.strip()
    d = cohort[["subject_id", "y"]].merge(oasis, on="subject_id", how="left")
    rows = {}
    for label, g in (("Nondemented", d[d["y"] == 0]), ("Demented", d[d["y"] == 1])):
        cdr = g["CDR"].value_counts()
        rows[label] = {
            "n": str(len(g)),
            "Female, n (%)": f"{(g['M/F'] == 'F').sum()} ({100 * (g['M/F'] == 'F').mean():.1f})",
            "Age, mean ± SD (range)": f"{g['Age'].mean():.1f} ± {g['Age'].std():.1f} ({g['Age'].min()}–{g['Age'].max()})",
            "Aged under 60, n": str((g["Age"] < 60).sum()),
            "Education (1–5), mean ± SD": f"{g['Educ'].mean():.2f} ± {g['Educ'].std():.2f}",
            "MMSE, mean ± SD": f"{g['MMSE'].mean():.1f} ± {g['MMSE'].std():.1f}",
            "CDR 0 / 0.5 / 1 / 2, n": " / ".join(str(int(cdr.get(v, 0))) for v in (0, 0.5, 1, 2)),
        }
    return pd.DataFrame(rows)


def main_table(y, folds, scores, probs, n_boot, seed):
    rows = {}
    for name in MODELS:
        s = scores[name]
        per_fold = [roc_auc_score(y[folds == k], s[folds == k]) for k in np.unique(folds)]
        row = {"Params": PARAMS[name], "AUC": f"{roc_auc_score(y, s):.4f}", "95% CI": bootstrap_ci(y, s, n_boot, seed),
               "Fold AUC": f"{np.mean(per_fold):.3f} ± {np.std(per_fold, ddof=1):.3f}"}
        if name in probs:
            p = probs[name]
            pred = (p >= 0.5).astype(int)
            tp, tn = ((pred == 1) & (y == 1)).sum(), ((pred == 0) & (y == 0)).sum()
            logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
            cal = LogisticRegression(penalty=None).fit(logit[:, None], y)
            row.update({"Acc.": f"{(tp + tn) / len(y):.3f}", "Sens.": f"{tp / y.sum():.3f}",
                        "Spec.": f"{tn / (len(y) - y.sum()):.3f}", "Brier": f"{brier_score_loss(y, p):.3f}",
                        "Slope": fmt2(cal.coef_[0][0]), "Intercept": fmt2(cal.intercept_[0])})
        else:
            row.update({k: "n/a" for k in ("Acc.", "Sens.", "Spec.", "Brier", "Slope", "Intercept")})
        rows[name] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def primary_table(cohort, y, scores, n_boot, seed):
    """F1: MMSE (and LR-1) vs each model with imaging inputs, Holm; F2: LR-3 vs LR-1; F3: LR-3 vs ANFIS."""
    p_mmse = [delong_p(y, scores["MMSE alone"], scores[m]) for m in IMAGING_INPUT]
    p_lr1 = [delong_p(y, scores["LR-1"], scores[m]) for m in IMAGING_INPUT]
    f1 = pd.DataFrame({
        "dAUC vs MMSE": [f"{roc_auc_score(y, scores[m]) - roc_auc_score(y, scores['MMSE alone']):+.3f}" for m in IMAGING_INPUT],
        "p": [fmt_p(p) for p in p_mmse], "Holm p": [fmt_p(p) for p in holm(p_mmse)],
        "Holm p (LR-1)": [fmt_p(p) for p in holm(p_lr1)]}, index=IMAGING_INPUT)
    pairs = {}
    for label, mask in (("all", np.ones(len(y), bool)), ("aged 60+", cohort["age"].to_numpy() >= 60)):
        yy = y[mask]
        sub = {k: v[mask] for k, v in scores.items()}
        pairs[label] = {
            "LR-3 vs LR-1": {"dAUC": f"{roc_auc_score(yy, sub['LR-3']) - roc_auc_score(yy, sub['LR-1']):+.3f}",
                             "95% CI": bootstrap_delta_ci(yy, sub["LR-3"], sub["LR-1"], n_boot, seed),
                             "p": fmt_p(delong_p(yy, sub["LR-3"], sub["LR-1"])),
                             "likelihood ratio": likelihood_ratio(cohort, mask)},
            "LR-3 vs ANFIS-on-GT": {"dAUC": f"{roc_auc_score(yy, sub['LR-3']) - roc_auc_score(yy, sub['ANFIS-on-GT']):+.3f}",
                                    "95% CI": bootstrap_delta_ci(yy, sub["LR-3"], sub["ANFIS-on-GT"], n_boot, seed),
                                    "p": fmt_p(delong_p(yy, sub["LR-3"], sub["ANFIS-on-GT"]))},
        }
    deep = list(DEEP_COLUMNS)
    p_deep = [delong_p(y, scores["ANFIS-on-GT"], scores[m]) for m in deep]
    exploratory = {m: {"p": fmt_p(p), "Holm p": fmt_p(h)} for m, p, h in zip(deep, p_deep, holm(p_deep))}
    exploratory["CART-3"] = {"p": fmt_p(delong_p(y, scores["ANFIS-on-GT"], scores["CART-3"]))}
    return f1, pairs, exploratory


def age_table(cohort, y, scores):
    old = cohort["age"].to_numpy() >= 60
    p = [delong_p(y[old], scores["MMSE alone"][old], scores[m][old]) for m in IMAGING_INPUT]
    p_by, h_by = dict(zip(IMAGING_INPUT, p)), dict(zip(IMAGING_INPUT, holm(p)))
    rows = {m: {"AUC (all)": f"{roc_auc_score(y, scores[m]):.3f}", "AUC (60+)": f"{roc_auc_score(y[old], scores[m][old]):.3f}",
                "p vs MMSE": fmt_p(p_by[m]) if m in p_by else "---", "Holm p": fmt_p(h_by[m]) if m in h_by else "---"}
            for m in MODELS}
    return pd.DataFrame.from_dict(rows, orient="index"), int(old.sum()), int(y[old].sum())


def epv_table(lr):
    rows = {}
    for name, params in (("LR-3", 4), ("CART-3", 7), ("ANFIS", 126)):
        n_param, n_risk = riley_n(params, lr["r2cs"], lr["max_r2cs"], EVENTS / 226)
        rows[name] = {"Params": params, "EPV": f"{EVENTS / params:.2f}", "Riley n": n_param}
    return pd.DataFrame.from_dict(rows, orient="index"), n_risk


# ---------------------------------------------------------------- figures

def auc3(y, s):
    """Three decimals, rounding the four-decimal value half up as the text does (0.7705 -> 0.771)."""
    return str(Decimal(f"{roc_auc_score(y, s):.4f}").quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def _sans_serif():
    """Arial where installed (Windows, macOS), otherwise a metric-compatible or default sans serif."""
    from matplotlib import font_manager
    installed = {f.name for f in font_manager.fontManager.ttflist}
    return next((f for f in ("Arial", "Liberation Sans", "Helvetica") if f in installed), "DejaVu Sans")


def style():
    plt.rcParams.update({
        "font.family": _sans_serif(), "font.size": 7.5, "axes.labelsize": 8,
        "legend.fontsize": 8, "legend.title_fontsize": 8, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "pdf.fonttype": 42, "lines.solid_capstyle": "butt",
    })


def plot_roc(y, scores, path):
    fig, ax = plt.subplots(figsize=(3.5, 3.3))
    ax.plot([0, 1], [0, 1], color="0.55", lw=0.6, ls=":", zorder=1)
    deep_aucs = [auc3(y, scores[m]) for m in DEEP_COLUMNS]
    for i, name in enumerate(DEEP_COLUMNS):
        fpr, tpr, _ = roc_curve(y, scores[name])
        ax.plot(fpr, tpr, color=GREY, lw=0.6, alpha=0.9, zorder=2,
                label=f"Deep ×5 ({min(deep_aucs)}–{max(deep_aucs)})" if i == 0 else None)
    curves = [  # drawn bottom to top; the legend lists them by AUC
        ("LR-2", "LR-2, imaging only", BLUE, 1.1, "-"),
        ("ANFIS-on-extracted", "ANFIS-on-extracted", PURPLE, 1.1, DASH_DOT),
        ("ANFIS-on-GT", "ANFIS-on-GT", VERMILION, 1.3, DASH_DOT),
        ("CART-3", "CART-3", ORANGE, 1.0, (0, (2.5, 1.5))),
        ("LR-1", "LR-1", SKY, 1.1, (0, (5, 2))),
        ("LR-3", "LR-3", GREEN, 1.3, "-"),
        ("MMSE alone", "MMSE alone", BLACK, 1.6, "-"),
    ]
    for name, label, color, lw, ls in curves:
        fpr, tpr, _ = roc_curve(y, scores[name])
        ax.plot(fpr, tpr, color=color, lw=lw, ls=ls, zorder=3, label=f"{label} ({auc3(y, scores[name])})")
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: -float(labels[i].rsplit("(", 1)[1].split("–")[-1].rstrip(")")))
    ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower right", title="Model (AUC)",
              frameon=True, framealpha=0.95, edgecolor="0.75", fancybox=False, borderpad=0.4,
              labelspacing=0.25, handlelength=2.4, handletextpad=0.5, borderaxespad=0.3)
    ax.set(xlim=(-0.01, 1.01), ylim=(-0.01, 1.01), xlabel="False positive rate", ylabel="True positive rate")
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
    fig.tight_layout(pad=0.3)
    fig.savefig(path)
    plt.close(fig)


def plot_calibration(y, probs, path):
    fig, ax = plt.subplots(figsize=(3.5, 3.1))
    ax.plot([0, 1], [0, 1], color=BLACK, lw=0.7, ls=":", label="Perfect calibration", zorder=1)
    for name, color, ls in [("LR-3", GREEN, "-"), ("CART-3", ORANGE, (0, (2.5, 1.5))), ("ANFIS-on-GT", VERMILION, DASH_DOT),
                            ("LR-2", BLUE, "-"), ("ResNet-50", GREY, "-")]:
        observed, predicted = calibration_curve(y, probs[name], n_bins=6, strategy="quantile")
        ax.plot(predicted, observed, color=color, lw=1.1, ls=ls, marker="o", ms=3, zorder=2,
                label=f"{name} ({brier_score_loss(y, probs[name]):.3f})")
    ax.legend(loc="lower right", title="Model (Brier score)", frameon=True, framealpha=0.95, edgecolor="0.75",
              fancybox=False, borderpad=0.4, labelspacing=0.25, handlelength=2.2, handletextpad=0.5, borderaxespad=0.3)
    ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02), xlabel="Mean predicted probability",
           ylabel="Observed fraction demented")
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
    fig.tight_layout(pad=0.3)
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------- comparison with the paper

def check_against_paper(results):
    if PAPER is None:
        print("\nNo paper_values.json found; skipping the comparison with the paper.")
        return True
    mismatches = []

    def walk(expected, got, path):
        if isinstance(expected, dict):
            for key, value in expected.items():
                walk(value, got.get(key) if isinstance(got, dict) else None, f"{path}/{key}")
        elif expected != got:
            mismatches.append((path, got, expected))

    expected = {k: v for k, v in PAPER.items() if k in results}  # the cohort table needs --demographics
    walk(expected, results, "")
    total = sum(1 for _ in _leaves(expected))
    print(f"\n{total - len(mismatches)}/{total} reported values identical to the paper.")
    for path, got, want in mismatches:
        print(f"   {path}: reproduced {got}, paper {want}")
    return not mismatches


def _leaves(d):
    for v in d.values():
        if isinstance(v, dict):
            yield from _leaves(v)
        else:
            yield v


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--demographics", type=Path)
    parser.add_argument("--data", type=Path, default=REPO / "data")
    parser.add_argument("--out", type=Path, default=REPO / "results")
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cohort, y, folds, scores, probs = load(args.cohort, args.data)
    assert y.sum() == EVENTS and len(y) == 226, "unexpected cohort"
    results = {}
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        if args.demographics:
            t = cohort_table(cohort, args.demographics)
            print("Cohort\n", t.to_string())
            t.to_csv(args.out / "table_cohort.csv")
            results["cohort"] = t.to_dict()
        main = main_table(y, folds, scores, probs, args.n_boot, args.seed)
        print("\nAll models\n", main.to_string())
        main.to_csv(args.out / "table_models.csv", index_label="Model")
        results["models"] = main.to_dict(orient="index")

        f1, pairs, exploratory = primary_table(cohort, y, scores, args.n_boot, args.seed)
        print("\nF1: MMSE alone vs each model with imaging inputs (Holm within the family)\n", f1.to_string())
        print("\nF2 and F3\n", json.dumps(pairs, indent=1))
        print("\nExploratory: ANFIS-on-GT vs the deep networks\n", json.dumps(exploratory))
        f1.to_csv(args.out / "table_primary.csv", index_label="Model")
        results.update({"F1": f1.to_dict(orient="index"), "pairs": pairs, "exploratory": exploratory})

        epv, n_risk = epv_table(pairs["all"]["LR-3 vs LR-1"]["likelihood ratio"])
        print(f"\nEvents per variable (Riley n: criteria i-ii; criterion iii needs {n_risk} for any model)\n", epv.to_string())
        results["epv"] = {"table": epv.astype(str).to_dict(orient="index"), "riley_iii": str(n_risk)}

        cutoffs = {}
        for cut in (23, 26):  # conventional MMSE cut-offs, as an operating point for the unfitted score
            flagged = (cohort["mmse"] <= cut).to_numpy()
            cutoffs[f"MMSE <= {cut}"] = {"Sens.": f"{flagged[y == 1].mean():.3f}", "Spec.": f"{1 - flagged[y == 0].mean():.3f}"}
        split_nodes = []
        for k in np.unique(folds):
            train = cohort[cohort["fold"] != k]
            x = pd.DataFrame({"hippo": residualise_on_age(train, train, "aseg_hippo_total")[0],
                              "wbv": residualise_on_age(train, train, "wbv_final")[0], "mmse": train["mmse"]})
            tree = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=0)
            tree.fit(StandardScaler().fit_transform(x), train["y"])
            split_nodes.append(str(tree.tree_.node_count - tree.get_n_leaves()))
        print("\nMMSE cut-offs", cutoffs, "| CART-3 split nodes per fold", split_nodes)
        results.update({"mmse_cutoffs": cutoffs, "cart_split_nodes": " ".join(split_nodes)})

        age, n_old, events_old = age_table(cohort, y, scores)
        print(f"\nAged 60 or over (n = {n_old}, {events_old} demented)\n", age.to_string())
        age.to_csv(args.out / "table_age60.csv", index_label="Model")
        results["age60"] = age.to_dict(orient="index")

    style()
    plot_roc(y, scores, args.out / "fig_roc.pdf")
    plot_calibration(y, probs, args.out / "fig_calib.pdf")
    (args.out / "results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nTables, figures and results.json written to {args.out}")
    if not check_against_paper(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
