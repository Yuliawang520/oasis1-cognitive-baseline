"""The five slice-based deep networks of the benchmark, re-run with fixed seeds on OASIS-1 and ADNI1.

One module reproduces notebooks/alzheimer_baseline_{densenet121,vgg16,resnet50,deit,swin}_oasis_colab.ipynb
so that the networks can be trained with (1) explicit seeds, (2) the ADNI1 cohort on its shared five-fold
split and (3) a training pool subsampled to the OASIS-1 size (learning curve). Everything else is the
original pipeline: 24 coronal slices per subject (PNG, 8 bit), ImageNet-pretrained backbones, the 85/15
subject-level validation split by md5 hash, two phases (head only, then all layers at a lower rate),
inverse-frequency class weights (times 1.5 for the minority class in Swin-T), early stopping on the raw
validation accuracy (balanced accuracy for Swin-T), AMP, gradient clipping, batch size 32, and
subject-level averaging of the slice probabilities.

Differences from the original notebooks, all deliberate:
  * seeds: random, numpy and torch are seeded, cuDNN runs deterministically (benchmark off), and
    torch.use_deterministic_algorithms(True, warn_only=True) is requested. Results are reproducible on
    the same hardware, library versions and worker count (NUM_WORKERS = 4);
  * data source: the cohort table gives subject_id, label and fold; slices are read from a local folder;
  * pool_size: an optional stratified subsample of the training pool (train plus validation subjects) per
    fold, drawn with a seeded generator before the hash split, for the learning curve;
  * outputs: one out-of-fold CSV per (cohort, model, seed, pool) plus a JSON log per fold with the best
    validation metric, epochs, timing, library versions and GPU name. Checkpoints stay local and are
    deleted after the fold. Runs whose fold CSV exists are skipped, so a Colab session can resume.

Usage (Colab notebooks call run_matrix; the CLI is for local smoke tests):
    python scripts/slice_networks.py --cohort-csv COHORT --cohort-name oasis|adni --slices-dir SLICES \
        --out-dir OUT --models all --seeds 0 1 2 --pool-sizes full 181 [--folds 0 1] [--smoke]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader, Dataset

CLASS_ORDER = ["Nondemented", "Demented"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
N_SLICES = 24

BATCH_SIZE = 32
NUM_WORKERS = 4
PHASE1_EPOCHS = 12
PHASE2_EPOCHS = 15
PATIENCE = 6
MIN_DELTA = 0.001
PHASE1_LR = 1e-4
PHASE2_LR_CNN = 1e-6
PHASE2_LR_HEAD = 1e-5
WEIGHT_DECAY = 1e-4
VAL_PERCENT = 15  # subjects whose md5 hash falls below this value form the validation set
OASIS_POOL_SIZE = 181  # mean training pool of the OASIS-1 folds (226 subjects, five folds)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()


# ----------------------------------------------------------------------------- models
@dataclass(frozen=True)
class Spec:
    name: str            # key used in file names and on the command line
    title: str           # name used in the paper
    column: str          # column of data/oof_predictions.csv that holds the original run
    head_key: str        # substring of the parameter names that belong to the classification head
    balanced_es: bool    # early stopping on balanced accuracy (Swin-T) instead of raw accuracy
    minority_boost: float
    build: Callable[[], nn.Module]


def _densenet121() -> nn.Module:
    import torchvision
    from torchvision.models import DenseNet121_Weights
    model = torchvision.models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(model.classifier.in_features, 2))
    return model


def _vgg16() -> nn.Module:
    import torchvision
    from torchvision.models import VGG16_Weights
    model = torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)
    return model


def _resnet50() -> nn.Module:
    import torchvision
    from torchvision.models import ResNet50_Weights
    model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def _deit_small() -> nn.Module:
    import timm
    return timm.create_model("deit_small_patch16_224.fb_in1k", pretrained=True, num_classes=2)


def _swin_t() -> nn.Module:
    import torchvision
    from torchvision.models import Swin_T_Weights
    model = torchvision.models.swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
    model.head = nn.Linear(model.head.in_features, 2)
    return model


SPECS = {
    "densenet121": Spec("densenet121", "DenseNet-121", "densenet121", "classifier", False, 1.0, _densenet121),
    "vgg16": Spec("vgg16", "VGG-16", "vgg16", "classifier.6", False, 1.0, _vgg16),
    "resnet50": Spec("resnet50", "ResNet-50", "resnet50", "fc", False, 1.0, _resnet50),
    "deit_small": Spec("deit_small", "DeiT-Small", "deit_small", "head", False, 1.0, _deit_small),
    "swin_t": Spec("swin_t", "Swin-T", "swin_t", "head", True, 1.5, _swin_t),
}


# ----------------------------------------------------------------------------- determinism
def seed_everything(seed: int) -> None:
    """Seed every generator in use. CUBLAS_WORKSPACE_CONFIG must be set before CUDA is initialised;
    the notebooks export it in their first cell, this call only fills it in if it is still unset."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:  # very old torch without warn_only
        pass


def _worker_init(worker_id: int) -> None:
    seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(seed)
    random.seed(seed)


# ----------------------------------------------------------------------------- data
def load_cohort(cohort_csv: Path) -> pd.DataFrame:
    """subject_id, label ('Demented'/'Nondemented'), fold. Accepts the OASIS-1 ANFIS-on-GT out-of-fold file
    (columns subject_id, label, oof_fold) or the ADNI cohort file (subject_id, label, fold)."""
    df = pd.read_csv(cohort_csv)
    if "oof_fold" in df.columns and "fold" not in df.columns:
        df = df.rename(columns={"oof_fold": "fold"})
    df = df[["subject_id", "label", "fold"]].copy()
    df["subject_id"] = df["subject_id"].astype(str)
    assert set(df["label"].unique()) <= set(CLASS_ORDER), f"labels must be in {CLASS_ORDER}"
    return df


def build_manifest(slices_dir: Path, cohort: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid, label, fold in cohort[["subject_id", "label", "fold"]].itertuples(index=False):
        pngs = sorted((slices_dir / sid).glob("z*.png"))[:N_SLICES]
        for p in pngs:
            rows.append({"subject_id": sid, "path": str(p), "label": label, "fold": int(fold)})
    manifest = pd.DataFrame(rows)
    found = manifest["subject_id"].nunique() if len(manifest) else 0
    print(f"{len(manifest)} slices for {found}/{len(cohort)} subjects in {slices_dir}")
    short = manifest.groupby("subject_id").size()
    if (short < N_SLICES).any():
        print(f"  {int((short < N_SLICES).sum())} subjects have fewer than {N_SLICES} slices")
    return manifest


def build_transforms(is_train: bool):
    import torchvision.transforms as transforms
    norm = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    if is_train:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.0),
            transforms.ToTensor(),
            norm,
        ])
    return transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), norm])


class SliceDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, is_train: bool):
        self.rows = rows.reset_index(drop=True)
        self.transform = build_transforms(is_train)
        self.class_to_idx = {c: i for i, c in enumerate(CLASS_ORDER)}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        return self.transform(img), self.class_to_idx[row["label"]], row["subject_id"]


def collate_fn(batch):
    return (torch.stack([b[0] for b in batch]), torch.tensor([b[1] for b in batch]), [b[2] for b in batch])


def is_val_subject(sid: str) -> bool:
    """The notebooks' deterministic 85/15 split of the training pool."""
    return int(hashlib.md5(f"{sid}_valsplit".encode()).hexdigest(), 16) % 100 < VAL_PERCENT


def subsample_pool(pool: pd.DataFrame, pool_size: int | None, seed: int, fold: int) -> pd.DataFrame:
    """Stratified subsample of the training pool (one row per subject) to pool_size subjects."""
    if pool_size is None or len(pool) <= pool_size:
        return pool
    rng = np.random.RandomState(seed * 100 + fold)
    parts = []
    for label, grp in pool.groupby("label"):
        k = int(round(pool_size * len(grp) / len(pool)))
        parts.append(grp.iloc[rng.permutation(len(grp))[:k]])
    return pd.concat(parts).sort_values("subject_id").reset_index(drop=True)


# ----------------------------------------------------------------------------- training
class EarlyStopping:
    def __init__(self, patience: int, min_delta: float, path: Path):
        self.patience, self.min_delta, self.path = patience, min_delta, path
        self.best, self.counter = 0.0, 0

    def step(self, metric: float, model: nn.Module) -> bool:
        if metric > self.best + self.min_delta:
            self.best, self.counter = metric, 0
            torch.save(model.state_dict(), self.path)
            return False
        self.counter += 1
        return self.counter >= self.patience


def _autocast():
    return torch.autocast(device_type=DEVICE.type, enabled=USE_AMP)


def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels, _ in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        with _autocast():
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item() * labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion):
    """Returns loss, raw accuracy and balanced accuracy on the slice level."""
    model.eval()
    total_loss, total = 0.0, 0
    all_labels, all_preds = [], []
    for imgs, labels, _ in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        with _autocast():
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        all_labels.append(labels.cpu())
        all_preds.append(outputs.argmax(1).cpu())
        total += labels.size(0)
        total_loss += loss.item() * labels.size(0)
    y = torch.cat(all_labels).numpy()
    p = torch.cat(all_preds).numpy()
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    balanced = 0.5 * (tp / (tp + fn + 1e-9) + tn / (tn + fp + 1e-9))
    return total_loss / total, float((p == y).mean()), float(balanced)


@torch.no_grad()
def collect_predictions(model, loader):
    model.eval()
    labels_all, probs_all, sids_all = [], [], []
    for imgs, labels, sids in loader:
        with _autocast():
            outputs = model(imgs.to(DEVICE))
        probs_all.append(torch.softmax(outputs.float(), 1).cpu().numpy())
        labels_all.extend(labels.numpy())
        sids_all.extend(sids)
    return np.array(labels_all), np.vstack(probs_all), sids_all


def subject_level_aggregate(labels_arr, probs_arr, sids):
    sid_probs, sid_labels = defaultdict(list), {}
    for i, sid in enumerate(sids):
        sid_probs[sid].append(probs_arr[i])
        sid_labels[sid] = labels_arr[i]
    ids = list(sid_probs.keys())
    return np.array([sid_labels[s] for s in ids]), np.array([np.mean(sid_probs[s], axis=0) for s in ids]), ids


def run_fold(spec: Spec, manifest: pd.DataFrame, test_fold: int, seed: int, pool_size: int | None,
             work_dir: Path, epochs: tuple[int, int] = (PHASE1_EPOCHS, PHASE2_EPOCHS)) -> tuple[list[dict], dict]:
    """Train on the other folds (minus the hashed validation subjects) and predict the test fold."""
    seed_everything(seed)
    t0 = time.time()
    subjects = manifest.drop_duplicates("subject_id")[["subject_id", "label", "fold"]]
    test_subj = subjects.loc[subjects["fold"] == test_fold, "subject_id"]
    pool = subsample_pool(subjects[subjects["fold"] != test_fold].reset_index(drop=True), pool_size, seed, test_fold)
    val_subj = [s for s in pool["subject_id"] if is_val_subject(s)]
    train_subj = [s for s in pool["subject_id"] if not is_val_subject(s)]

    train_rows = manifest[manifest["subject_id"].isin(train_subj)]
    val_rows = manifest[manifest["subject_id"].isin(val_subj)]
    test_rows = manifest[manifest["subject_id"].isin(test_subj)]
    print(f"  fold {test_fold}: train {len(train_subj)} / val {len(val_subj)} / test {len(test_subj)} subjects"
          f" (pool {len(pool)}, {int((pool['label'] == 'Demented').sum())} Demented)")

    gen = torch.Generator().manual_seed(seed)
    ldr = dict(num_workers=NUM_WORKERS, persistent_workers=NUM_WORKERS > 0, pin_memory=DEVICE.type == "cuda",
               collate_fn=collate_fn, worker_init_fn=_worker_init)
    train_loader = DataLoader(SliceDataset(train_rows, True), batch_size=BATCH_SIZE, shuffle=True, generator=gen, **ldr)
    val_loader = DataLoader(SliceDataset(val_rows, False), batch_size=BATCH_SIZE * 2, shuffle=False, **ldr)
    test_loader = DataLoader(SliceDataset(test_rows, False), batch_size=BATCH_SIZE * 2, shuffle=False, **ldr)

    train_labels = train_rows["label"].map({"Nondemented": 0, "Demented": 1})
    counts = train_labels.value_counts()
    weights = torch.tensor([len(train_labels) / (2 * counts.get(0, 1)),
                            len(train_labels) / (2 * counts.get(1, 1)) * spec.minority_boost],
                           dtype=torch.float32, device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)

    model = spec.build().to(DEVICE)
    ckpt = work_dir / f"{spec.name}_seed{seed}_pool{pool_size or 'full'}_fold{test_fold}.pth"
    stopper = EarlyStopping(PATIENCE, MIN_DELTA, ckpt)
    history: list[dict] = []
    metric_of = (lambda acc, bal: bal) if spec.balanced_es else (lambda acc, bal: acc)

    # phase 1: head only
    for name, p in model.named_parameters():
        p.requires_grad = spec.head_key in name
    opt1 = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=PHASE1_LR, weight_decay=WEIGHT_DECAY)
    sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=epochs[0], eta_min=1e-6)
    scaler = torch.amp.GradScaler(DEVICE.type, enabled=USE_AMP)
    for epoch in range(1, epochs[0] + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, opt1, scaler)
        vl_loss, vl_acc, vl_bal = validate(model, val_loader, criterion)
        sch1.step()
        history.append(dict(phase=1, epoch=epoch, tr_loss=tr_loss, tr_acc=tr_acc, vl_loss=vl_loss, vl_acc=vl_acc, vl_bal=vl_bal))
        if stopper.step(metric_of(vl_acc, vl_bal), model):
            break

    # phase 2: all layers, backbone at the lower rate
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    for p in model.parameters():
        p.requires_grad = True
    stopper.counter = 0
    backbone = [p for n, p in model.named_parameters() if spec.head_key not in n]
    head = [p for n, p in model.named_parameters() if spec.head_key in n]
    opt2 = torch.optim.AdamW([{"params": backbone, "lr": PHASE2_LR_CNN}, {"params": head, "lr": PHASE2_LR_HEAD}],
                             weight_decay=WEIGHT_DECAY)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=epochs[1], eta_min=1e-8)
    scaler = torch.amp.GradScaler(DEVICE.type, enabled=USE_AMP)
    for epoch in range(1, epochs[1] + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, opt2, scaler)
        vl_loss, vl_acc, vl_bal = validate(model, val_loader, criterion)
        sch2.step()
        history.append(dict(phase=2, epoch=epoch, tr_loss=tr_loss, tr_acc=tr_acc, vl_loss=vl_loss, vl_acc=vl_acc, vl_bal=vl_bal))
        if stopper.step(metric_of(vl_acc, vl_bal), model):
            break

    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    labels_arr, probs_arr, sids = collect_predictions(model, test_loader)
    subj_labels, subj_probs, subj_ids = subject_level_aggregate(labels_arr, probs_arr, sids)
    rows = [{"subject_id": sid, "label": CLASS_ORDER[int(lbl)], "prob_demented": float(pr[1]), "fold": test_fold}
            for sid, lbl, pr in zip(subj_ids, subj_labels, subj_probs)]
    log = {
        "model": spec.name, "seed": seed, "pool_size": pool_size, "fold": test_fold,
        "n_train": len(train_subj), "n_val": len(val_subj), "n_test": len(test_subj),
        "best_val_metric": stopper.best, "metric": "balanced_accuracy" if spec.balanced_es else "accuracy",
        "epochs_run": len(history), "history": history, "seconds": round(time.time() - t0, 1),
        "fold_auc": float(roc_auc_score(subj_labels, subj_probs[:, 1])) if len(set(subj_labels)) == 2 else None,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor(),
        "torch": torch.__version__, "python": platform.python_version(),
    }
    ckpt.unlink(missing_ok=True)
    del model, opt1, opt2
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows, log


def run_matrix(cohort_name: str, cohort_csv: Path, slices_dir: Path, out_dir: Path, models: list[str],
               seeds: list[int], pool_sizes: list[int | None], folds: list[int] | None = None,
               epochs: tuple[int, int] = (PHASE1_EPOCHS, PHASE2_EPOCHS), work_dir: Path | None = None) -> pd.DataFrame:
    """Train every (model, seed, pool) on every fold, skipping folds whose CSV already exists."""
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = work_dir or out_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    cohort = load_cohort(cohort_csv)
    manifest = build_manifest(slices_dir, cohort)
    all_folds = sorted(manifest["fold"].unique()) if folds is None else folds
    versions = {"torch": torch.__version__}
    try:
        import torchvision
        versions["torchvision"] = torchvision.__version__
    except ImportError:
        pass
    try:
        import timm
        versions["timm"] = timm.__version__
    except ImportError:
        pass
    print("library versions:", versions, "| device:", DEVICE)

    summary = []
    for model_name in models:
        spec = SPECS[model_name]
        for pool_size in pool_sizes:
            for seed in seeds:
                tag = f"{cohort_name}_{spec.name}_seed{seed}_pool{pool_size or 'full'}"
                print(f"\n=== {tag} ===")
                rows_all = []
                for f in all_folds:
                    fold_csv = out_dir / f"oof_{tag}_fold{f}.csv"
                    if fold_csv.exists():
                        rows_all.append(pd.read_csv(fold_csv))
                        print(f"  fold {f}: found, skipped")
                        continue
                    rows, log = run_fold(spec, manifest, int(f), seed, pool_size, work_dir, epochs)
                    df = pd.DataFrame(rows)
                    df.to_csv(fold_csv, index=False)
                    (out_dir / f"log_{tag}_fold{f}.json").write_text(json.dumps({**log, **versions}, indent=1))
                    rows_all.append(df)
                    print(f"  fold {f}: AUC {log['fold_auc']:.3f}, best val {log['best_val_metric']:.3f}, "
                          f"{log['epochs_run']} epochs, {log['seconds']} s")
                oof = pd.concat(rows_all).sort_values("subject_id").reset_index(drop=True)
                oof.to_csv(out_dir / f"oof_{tag}.csv", index=False)
                y = (oof["label"] == "Demented").astype(int)
                auc = roc_auc_score(y, oof["prob_demented"]) if y.nunique() == 2 else float("nan")
                summary.append({"cohort": cohort_name, "model": spec.name, "seed": seed, "pool_size": pool_size or "full",
                                "n": len(oof), "auc": round(float(auc), 4)})
                print(f"  pooled out-of-fold AUC: {auc:.4f}")
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(out_dir / f"summary_{cohort_name}.csv", index=False)
    print("\n" + summary_df.to_string(index=False))
    return summary_df


def seed_ensemble(out_dir: Path, cohort_name: str, model_name: str, pool_size: int | None, seeds: list[int]) -> pd.DataFrame:
    """Mean of the seed probabilities per subject (the pre-specified primary output of a multi-seed run)."""
    frames = []
    for s in seeds:
        df = pd.read_csv(out_dir / f"oof_{cohort_name}_{model_name}_seed{s}_pool{pool_size or 'full'}.csv")
        frames.append(df.set_index("subject_id")["prob_demented"].rename(f"seed{s}"))
    probs = pd.concat(frames, axis=1)
    base = pd.read_csv(out_dir / f"oof_{cohort_name}_{model_name}_seed{seeds[0]}_pool{pool_size or 'full'}.csv")
    base = base.set_index("subject_id")[["label", "fold"]]
    out = base.join(probs)
    out["prob_demented"] = probs.mean(axis=1)
    return out.reset_index()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cohort-csv", type=Path, required=True)
    ap.add_argument("--cohort-name", choices=("oasis", "adni"), required=True)
    ap.add_argument("--slices-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--models", nargs="+", default=["all"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--pool-sizes", nargs="+", default=["full"], help="'full' or an integer such as 181")
    ap.add_argument("--folds", nargs="*", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="one epoch per phase, for a local code check")
    args = ap.parse_args()
    models = list(SPECS) if args.models == ["all"] else args.models
    pools = [None if p == "full" else int(p) for p in args.pool_sizes]
    epochs = (1, 1) if args.smoke else (PHASE1_EPOCHS, PHASE2_EPOCHS)
    if args.smoke:
        global NUM_WORKERS
        NUM_WORKERS = 0
    run_matrix(args.cohort_name, args.cohort_csv, args.slices_dir, args.out_dir, models, args.seeds, pools,
               args.folds, epochs)


if __name__ == "__main__":
    main()
