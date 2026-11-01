"""Pack the baseline T1 images of the ADNI network cohort for upload: one RAS-reoriented, gzipped NIfTI per
participant, named <subject_id>.nii.gz, exactly what notebooks/adni_slices_colab.ipynb would otherwise build on
Colab from the raw IDA download. Uploading this folder (about 12 GB for 601 participants) instead of the full
collection (70 GB) or the raw baseline files (22 GB) saves time; the pixel data are unchanged (float32, lossless
gzip, axis permutation only).

ADNI data cannot be redistributed; the output stays outside the repository.

Usage: python scripts/adni_pack_t1.py --cohort adni_network_cohort.csv --adni-root "D:/ADNI1_Complete 1Yr 1.5T/ADNI" --out adni_t1_prepared
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cohort", type=Path, required=True)
    ap.add_argument("--adni-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cohort = pd.read_csv(args.cohort)
    rows, t0 = [], time.time()
    for n, (sid, rel) in enumerate(zip(cohort["subject_id"].astype(str), cohort["image_relpath"]), 1):
        src = args.adni_root / rel
        dst = args.out / f"{sid}.nii.gz"
        if not dst.exists():
            img = nib.as_closest_canonical(nib.load(str(src)))
            nib.save(nib.Nifti1Image(np.asarray(img.dataobj, dtype=np.float32), img.affine), str(dst))
        rows.append({"subject_id": sid, "source": rel, "source_mb": round(os.path.getsize(src) / 1e6, 1),
                     "packed_mb": round(os.path.getsize(dst) / 1e6, 1)})
        if n % 50 == 0 or n == len(cohort):
            print(f"{n}/{len(cohort)} packed, {time.time() - t0:.0f} s", flush=True)
    manifest = pd.DataFrame(rows)
    manifest.to_csv(args.out / "packed_manifest.csv", index=False)
    print(f"{len(manifest)} files: {manifest['source_mb'].sum() / 1000:.1f} GB -> {manifest['packed_mb'].sum() / 1000:.1f} GB in {args.out}")


if __name__ == "__main__":
    main()
