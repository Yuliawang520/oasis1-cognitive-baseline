"""Coronal slice extraction for the ADNI1 replication of the five slice-based networks.

The OASIS-1 networks (notebooks/alzheimer_baseline_*_oasis_colab.ipynb) read FreeSurfer's brainmask.mgz,
a skull-stripped, intensity-normalised volume conformed to 1 mm, 256^3, LIA orientation, and took
vol[:, :, k] for k = linspace(int(0.27 * 256), int(0.47 * 256) - 1, 24) = 69 ... 119. In a conformed
FreeSurfer volume the third axis runs posterior -> anterior, so these are 24 CORONAL slices (the original
notebooks label them "axial"; the code, not the label, defines the pipeline). Each slice was min-max
scaled to 8 bit and saved as PNG.

ADNI1 has no brainmask.mgz. This module reproduces the geometry and the intensity convention:
  1. the ADNI preprocessed T1 (GradWarp, B1, N3, Scaled) is skull-stripped with a mask supplied by the
     caller (HD-BET on Colab; any binary mask in the T1 voxel grid works),
  2. the stripped image is conformed like mri_convert --conform (1 mm, 256^3, LIA, trilinear, FOV-centred),
  3. intensities are scaled so that the white-matter peak sits at 110 and clipped to 0 ... 255, the
     convention of FreeSurfer's T1.mgz from which brainmask.mgz is cut,
  4. the same 24 coronal slices are written as 8-bit PNGs with the same min-max rule,
  5. a QC table records the brain extent along the slice axis, the fraction of the 69 ... 119 window that
     lies inside the brain, the white-matter peak and the mask volume.

`--center brain` is a pre-specified fallback: it shifts the conformed grid (integer voxels, no resampling)
so that the brain's bounding-box centre sits at voxel 128 on every axis. It is used only if the QC of the
FOV-centred run shows the window leaving the brain in more than 10% of subjects (see
docs/prespecification_networks_adni.md).

ADNI data cannot be redistributed; this module writes nothing into the repository.

Usage (local or Colab):
    python scripts/adni_slices.py --cohort adni_network_cohort.csv --t1-dir T1_DIR --mask-dir MASK_DIR \
        --out-dir SLICES_DIR [--center fov|brain] [--limit N]
T1_DIR holds <subject_id>.nii.gz, MASK_DIR holds <subject_id>.nii.gz brain masks (HD-BET output names).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.processing import conform
from PIL import Image
from scipy.ndimage import gaussian_filter1d

CONFORM_SHAPE = (256, 256, 256)
CONFORM_ORIENTATION = "LIA"
SLICE_AXIS = 2                      # third axis of a conformed volume: posterior -> anterior
SLICE_RANGE = (0.27, 0.47)          # fraction of the axis length, as in the OASIS-1 notebooks
N_SLICES = 24
WM_TARGET = 110.0                   # FreeSurfer T1.mgz convention
MIN_SLICE_MAX = 1e-3                # OASIS-1 rule: a slice is skipped if its maximum is below this


def coronal_indices(n: int = CONFORM_SHAPE[SLICE_AXIS]) -> np.ndarray:
    lo, hi = SLICE_RANGE
    return np.linspace(int(n * lo), int(n * hi) - 1, N_SLICES, dtype=int)


def conform_like_freesurfer(img: nib.Nifti1Image, order: int) -> nib.Nifti1Image:
    """mri_convert --conform geometry: 1 mm isotropic, 256^3, LIA, centred on the input FOV centre."""
    return conform(img, out_shape=CONFORM_SHAPE, voxel_size=(1.0, 1.0, 1.0), order=order, cval=0.0,
                   orientation=CONFORM_ORIENTATION)


def brain_bbox(mask: np.ndarray) -> list[tuple[int, int]]:
    """(first, last) non-zero index along each axis."""
    out = []
    for axis in range(mask.ndim):
        other = tuple(a for a in range(mask.ndim) if a != axis)
        nz = np.where(mask.any(axis=other))[0]
        out.append((int(nz[0]), int(nz[-1])) if len(nz) else (-1, -1))
    return out


def center_on_brain(vol: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Integer shift so that the mask bounding-box centre sits at the volume centre (no resampling)."""
    shifts = []
    for axis, (lo, hi) in enumerate(brain_bbox(mask)):
        shift = vol.shape[axis] // 2 - (lo + hi) // 2
        shifts.append(int(shift))
    vol = np.roll(vol, shifts, axis=(0, 1, 2))
    mask = np.roll(mask, shifts, axis=(0, 1, 2))
    return vol, mask, shifts


def wm_peak(vol: np.ndarray, mask: np.ndarray) -> float:
    """Mode of the brain intensities above their median, from a smoothed 256-bin histogram.
    In a T1 image the white matter is the brightest large tissue class, so this is the WM peak."""
    brain = vol[mask > 0]
    brain = brain[brain > 0]
    upper = brain[brain >= np.median(brain)]
    hist, edges = np.histogram(upper, bins=256)
    smooth = gaussian_filter1d(hist.astype(float), sigma=3)
    k = int(np.argmax(smooth))
    return float(0.5 * (edges[k] + edges[k + 1]))


def normalise_like_t1mgz(vol: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float]:
    peak = wm_peak(vol, mask)
    out = np.clip(vol * (WM_TARGET / peak), 0.0, 255.0)
    out[mask == 0] = 0.0
    return out, peak


def slice_to_uint8(s: np.ndarray) -> np.ndarray | None:
    """The OASIS-1 notebooks' rule: skip near-empty slices, otherwise min-max to 0 ... 255."""
    if s.max() <= MIN_SLICE_MAX:
        return None
    s = (s - s.min()) / (s.max() - s.min() + 1e-8)
    return (s * 255).astype(np.uint8)


def process_subject(t1_path: Path, mask_path: Path, out_dir: Path, subject_id: str, center: str = "fov",
                    overwrite: bool = False) -> dict:
    """Strip, conform, normalise and slice one subject. Returns the QC record."""
    t1 = nib.load(str(t1_path))
    mask_img = nib.load(str(mask_path))
    t1_data = np.asarray(t1.dataobj, dtype=np.float32)
    mask_data = (np.asarray(mask_img.dataobj) > 0.5).astype(np.float32)
    if mask_data.shape != t1_data.shape:
        raise ValueError(f"{subject_id}: mask shape {mask_data.shape} differs from T1 shape {t1_data.shape}")
    stripped = nib.Nifti1Image(t1_data * mask_data, t1.affine)
    mask_native = nib.Nifti1Image(mask_data, t1.affine)

    conf = conform_like_freesurfer(stripped, order=1)          # trilinear, as mri_convert's default
    conf_mask = conform_like_freesurfer(mask_native, order=0)  # nearest neighbour for the mask
    vol = np.asarray(conf.dataobj, dtype=np.float32)
    mask = np.asarray(conf_mask.dataobj) > 0.5
    vol[~mask] = 0.0
    shifts = [0, 0, 0]
    if center == "brain":
        vol, mask, shifts = center_on_brain(vol, mask)
    elif center != "fov":
        raise ValueError("center must be 'fov' or 'brain'")

    vol, peak = normalise_like_t1mgz(vol, mask)
    bbox = brain_bbox(mask)
    idx = coronal_indices(vol.shape[SLICE_AXIS])
    k_lo, k_hi = bbox[SLICE_AXIS]
    inside = int(((idx >= k_lo) & (idx <= k_hi)).sum())

    subj_dir = out_dir / subject_id
    subj_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for i, k in enumerate(idx):
        png = subj_dir / f"z{i:02d}.png"
        if png.exists() and not overwrite:
            written += 1
            continue
        s = slice_to_uint8(np.take(vol, k, axis=SLICE_AXIS))
        if s is not None:
            Image.fromarray(s).save(png)
            written += 1
    voxel_ml = float(np.prod(mask_img.header.get_zooms()[:3])) / 1000.0
    return {
        "subject_id": subject_id, "n_png": written, "wm_peak_raw": round(peak, 2),
        "mask_volume_ml": round(float(mask_data.sum()) * voxel_ml, 1),
        "brain_k_first": k_lo, "brain_k_last": k_hi, "window_first": int(idx[0]), "window_last": int(idx[-1]),
        "window_inside_brain": inside / N_SLICES,
        "brain_i_extent": f"{bbox[0][0]}-{bbox[0][1]}", "brain_j_extent": f"{bbox[1][0]}-{bbox[1][1]}",
        "shift_ijk": json.dumps(shifts), "t1_shape": json.dumps(list(t1_data.shape)),
        "t1_zooms": json.dumps([round(float(z), 3) for z in t1.header.get_zooms()[:3]]),
    }


def find_mask(mask_dir: Path, subject_id: str) -> Path:
    """HD-BET names the mask of <sid>.nii.gz as <sid>_bet.nii.gz; a plain <sid>.nii.gz is accepted too."""
    for name in (f"{subject_id}_bet.nii.gz", f"{subject_id}.nii.gz", f"{subject_id}_mask.nii.gz"):
        if (mask_dir / name).exists():
            return mask_dir / name
    raise FileNotFoundError(f"no brain mask for {subject_id} in {mask_dir}")


def qc_montage(slices_dir: Path, subject_ids: list[str], path: Path, every: int = 3) -> None:
    """One row per subject with every third slice of the 24, for a visual check of plane and stripping."""
    rows = []
    for sid in subject_ids:
        tiles = []
        for i in range(0, N_SLICES, every):
            png = slices_dir / sid / f"z{i:02d}.png"
            tiles.append(np.asarray(Image.open(png)) if png.exists() else np.zeros((256, 256), np.uint8))
        rows.append(np.concatenate(tiles, axis=1))
    Image.fromarray(np.concatenate(rows, axis=0)).save(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cohort", type=Path, required=True, help="CSV with a subject_id column (adni_network_cohort.csv)")
    ap.add_argument("--t1-dir", type=Path, required=True, help="folder with <subject_id>.nii.gz T1 images")
    ap.add_argument("--mask-dir", type=Path, required=True, help="folder with <subject_id>.nii.gz brain masks")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--center", choices=("fov", "brain"), default="fov")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    subjects = pd.read_csv(args.cohort)["subject_id"].astype(str).tolist()
    if args.limit:
        subjects = subjects[: args.limit]
    records, failures = [], []
    for n, sid in enumerate(subjects, 1):
        try:
            records.append(process_subject(args.t1_dir / f"{sid}.nii.gz", find_mask(args.mask_dir, sid),
                                           args.out_dir, sid, args.center, args.overwrite))
        except Exception as exc:  # noqa: BLE001 - keep going, report at the end
            failures.append({"subject_id": sid, "error": repr(exc)})
        if n % 25 == 0 or n == len(subjects):
            print(f"{n}/{len(subjects)} subjects processed, {len(failures)} failures", flush=True)
    qc = pd.DataFrame(records)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    qc.to_csv(args.out_dir / f"qc_slices_{args.center}.csv", index=False)
    if failures:
        pd.DataFrame(failures).to_csv(args.out_dir / f"failures_{args.center}.csv", index=False)
    if len(qc):
        print(qc[["n_png", "wm_peak_raw", "mask_volume_ml", "brain_k_first", "brain_k_last",
                  "window_inside_brain"]].describe().round(2).to_string())
        outside = (qc["window_inside_brain"] < 1).mean()
        print(f"\nwindow leaves the brain in {outside:.1%} of subjects (pre-specified fallback threshold: 10%)")
        sample = qc["subject_id"].sample(min(8, len(qc)), random_state=0).tolist()
        qc_montage(args.out_dir, sample, args.out_dir / f"qc_montage_{args.center}.png")
    print(f"QC table: {args.out_dir / f'qc_slices_{args.center}.csv'}; failures: {len(failures)}")


if __name__ == "__main__":
    main()
