"""The ANFIS of notebooks/alzheimer_anfis_train_colab.ipynb, with switches for the ablations.

The defaults reproduce the paper's ANFIS: three Gaussian membership functions per input, a first-order
Takagi-Sugeno-Kang rule base, a class-weighted smooth-L1 regression on the CDR, SMOTE within the
training fold, and Adam for 300 full-batch epochs. Training is deterministic: the consequents start at
zero, the membership functions start at training-fold quantiles, and SMOTE has a fixed random state.
"""
import itertools
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsRegressor


@dataclass(frozen=True)
class Config:
    n_mfs: int = 3
    smote: bool = True
    smote_seed: int = 42
    class_weights: bool = True
    target: str = "cdr"  # "cdr": smooth-L1 regression on the CDR; "binary": logistic loss on the label
    epochs: int = 300
    lr: float = 5e-3
    weight_decay: float = 1e-4
    consequent_l2: float = 1e-2


def n_parameters(n_inputs, n_mfs):
    """First-order consequents (n_inputs + 1 per rule) plus a centre and a width per membership function."""
    return n_mfs ** n_inputs * (n_inputs + 1) + 2 * n_inputs * n_mfs


class GaussianMF(nn.Module):
    def __init__(self, centers, sigmas):
        super().__init__()
        self.centers = nn.Parameter(torch.tensor(centers, dtype=torch.float32))
        self.log_sigmas = nn.Parameter(torch.log(torch.tensor(sigmas, dtype=torch.float32) + 1e-6))

    def forward(self, x):
        return torch.exp(-0.5 * ((x.unsqueeze(-1) - self.centers) / self.log_sigmas.exp()) ** 2)


class ANFIS(nn.Module):
    """Fuzzification, product T-norm firing, normalisation, first-order TSK consequents, weighted sum."""

    def __init__(self, n_inputs, n_mfs, centers, sigmas):
        super().__init__()
        self.mf = GaussianMF(centers, sigmas)
        rules = list(itertools.product(range(n_mfs), repeat=n_inputs))
        self.register_buffer("rule_idx", torch.tensor(rules, dtype=torch.long))
        self.consequent = nn.Parameter(torch.zeros(len(rules), n_inputs + 1))

    def forward(self, x):
        mu = self.mf(x)
        idx = self.rule_idx.T.unsqueeze(0).expand(x.shape[0], -1, -1)
        firing = torch.gather(mu, 2, idx).prod(dim=1)
        norm_firing = firing / (firing.sum(dim=1, keepdim=True) + 1e-9)
        x_aug = torch.cat([x, torch.ones(x.shape[0], 1, dtype=x.dtype)], dim=1)
        return (norm_firing * (x_aug @ self.consequent.T)).sum(dim=1)


def init_mf_params(x, n_mfs):
    centers = np.zeros((x.shape[1], n_mfs))
    sigmas = np.zeros((x.shape[1], n_mfs))
    for i in range(x.shape[1]):
        centers[i] = np.percentile(x[:, i], np.linspace(20, 80, n_mfs))
        sigmas[i] = max(0.5 * (np.percentile(x[:, i], 66) - np.percentile(x[:, i], 33)), 1e-3)
    return centers, sigmas


def fit_predict(x_train, cdr_train, label_train, x_test, cfg=Config()):
    """Train on one fold (standardised float32 features) and return the raw output for the test fold:
    the CDR-scale severity for target "cdr", the logit for target "binary"."""
    x_res, cdr_res, label_res = x_train, cdr_train, label_train
    if cfg.smote:
        minority = min(np.bincount(label_train))
        k = max(1, min(5, minority - 1))
        if minority > k:
            x_res, label_res = SMOTE(k_neighbors=k, random_state=cfg.smote_seed).fit_resample(x_train, label_train)
            cdr_res = KNeighborsRegressor(n_neighbors=1).fit(x_train, cdr_train).predict(x_res)
    if cfg.class_weights:
        counts = np.bincount(label_res, minlength=2)
        weights = (len(label_res) / (2.0 * np.maximum(counts, 1)))[label_res]
    else:
        weights = np.ones(len(label_res))

    centers, sigmas = init_mf_params(x_res, cfg.n_mfs)
    model = ANFIS(x_res.shape[1], cfg.n_mfs, centers, sigmas)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.target == "cdr":
        loss_fn, target = nn.SmoothL1Loss(reduction="none"), cdr_res
    else:
        loss_fn, target = nn.BCEWithLogitsLoss(reduction="none"), label_res
    x_t = torch.tensor(x_res, dtype=torch.float32)
    t_t = torch.tensor(target, dtype=torch.float32)
    w_t = torch.tensor(weights, dtype=torch.float32)
    model.train()
    for _ in range(cfg.epochs):
        optimizer.zero_grad()
        loss = (loss_fn(model(x_t), t_t) * w_t).mean() + cfg.consequent_l2 * model.consequent.pow(2).sum()
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(x_test, dtype=torch.float32)).numpy()
