"""Build the cohort file that drives the ADNI1 network replication (kept outside the repository).

One row per participant of the tabular ADNI replication (scripts/adni_replication.py): the 601 ADNI1
"Complete 1Yr 1.5T" participants with a baseline FreeSurfer row. Columns: subject_id, site (first three
characters of the PTID), fold (the stratified five-fold split of the tabular replication), label
('Demented' if the global CDR at baseline is above 0, else 'Nondemented', the OASIS-1 label names the
training code expects), cdglobal, mmse, age, sex, and the baseline T1 image: image_id, description and
the path relative to the ADNI download root. Scanner manufacturer and model come from the ADNIMERGE2
MRIQC table (screening MPRAGE series) when available.

ADNI data cannot be redistributed: write the output next to the ADNI files, never into the repository.

Usage: python scripts/adni_network_cohort.py --clinical clinical_metadata.csv --folds results/adni/adni_folds.csv
           --manifest manifest_images.csv --adni-root "D:/ADNI1_Complete 1Yr 1.5T/ADNI"
           [--adnimerge2 path/to/ADNIMERGE2] --out adni_network_cohort.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clinical", type=Path, required=True)
    ap.add_argument("--folds", type=Path, required=True, help="adni_folds.csv written by adni_replication.py")
    ap.add_argument("--manifest", type=Path, required=True, help="manifest_images.csv of the longitudinal project")
    ap.add_argument("--adni-root", type=Path, required=True, help="the ADNI/ folder of the IDA download")
    ap.add_argument("--adnimerge2", type=Path, default=None, help="root of the extracted ADNIMERGE2 R package")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    clinical = pd.read_csv(args.clinical)
    bl = clinical[clinical["timepoint"] == "bl"].copy()
    bl["subject_id"] = bl["subject_id"].astype(str)
    folds = pd.read_csv(args.folds)
    folds["subject_id"] = folds["subject_id"].astype(str)
    cohort = folds.merge(bl[["subject_id", "cdglobal", "mmscore", "age_exact", "sex"]], on="subject_id",
                         how="left", validate="1:1")
    assert cohort[["cdglobal", "mmscore", "age_exact"]].notna().all().all(), "clinical data missing for some folds rows"
    cohort = cohort.rename(columns={"mmscore": "mmse", "age_exact": "age"})
    cohort["label"] = (cohort["cdglobal"] > 0).map({True: "Demented", False: "Nondemented"})
    cohort["site"] = cohort["subject_id"].str[:3]

    manifest = pd.read_csv(args.manifest)
    img = manifest[manifest["timepoint"] == "bl"][["subject_id", "image_id", "description", "path"]].copy()
    img["subject_id"] = img["subject_id"].astype(str)
    root = str(args.adni_root).replace("\\", "/").rstrip("/") + "/"
    img["image_relpath"] = img["path"].str.replace("\\", "/", regex=False).str.replace(root, "", regex=False)
    assert (img["image_relpath"] != img["path"]).all(), "manifest paths do not start with --adni-root"
    cohort = cohort.merge(img.drop(columns="path"), on="subject_id", how="left", validate="1:1")
    assert cohort["image_id"].notna().all(), "baseline image missing for some participants"

    cohort["manufacturer"] = pd.NA
    cohort["scanner_model"] = pd.NA
    if args.adnimerge2 is not None:
        # MRIQC lists the raw series (not the processed image IDs) with the scanner; take the screening
        # MPRAGE series of each participant and the most frequent manufacturer and model among them.
        import pyreadr  # noqa: E402 - optional dependency
        qc = list(pyreadr.read_r(str(args.adnimerge2 / "data" / "MRIQC.rda")).values())[0]
        qc["ParticipantID"] = qc["ParticipantID"].astype(str)
        sc = qc[qc["VISCODE2"].astype(str).isin(["sc", "scmri", "bl"])
                & qc["SeriesDescription"].astype(str).str.upper().str.contains("MP-?RAGE", regex=True)]

        def mode_or_na(s):
            s = s.dropna().astype(str).str.upper()
            return s.mode().iloc[0] if len(s) else pd.NA

        mode = (sc.groupby("ParticipantID")[["ScannerManufacturer", "ScannerModel"]].agg(mode_or_na)
                  .rename(columns={"ScannerManufacturer": "manufacturer", "ScannerModel": "scanner_model"}))
        cohort = cohort.drop(columns=["manufacturer", "scanner_model"]).merge(
            mode, left_on="subject_id", right_index=True, how="left")
        print("manufacturer:", cohort["manufacturer"].value_counts(dropna=False).to_dict())

    cohort = cohort[["subject_id", "site", "fold", "label", "cdglobal", "mmse", "age", "sex",
                     "image_id", "description", "image_relpath", "manufacturer", "scanner_model"]]
    cohort.to_csv(args.out, index=False)
    print(f"{len(cohort)} participants, {int((cohort['label'] == 'Demented').sum())} Demented, "
          f"folds {sorted(cohort['fold'].unique())}; written to {args.out}")


if __name__ == "__main__":
    main()
