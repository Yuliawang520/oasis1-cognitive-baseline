"""Build the cohort file that drives the ADNI1 network replication (kept outside the repository).

One row per participant of the tabular ADNI replication (scripts/adni_replication.py): the 601 ADNI1
"Complete 1Yr 1.5T" participants with a baseline FreeSurfer row. Columns: subject_id, site (first three
characters of the PTID), fold (the stratified five-fold split of the tabular replication), label
('Demented' if the global CDR at baseline is above 0, else 'Nondemented', the OASIS-1 label names the
training code expects), cdglobal, mmse, age, sex, and the baseline T1 image: image_id, description and
the path relative to the ADNI download root. Scanner manufacturer and model come from the IDA metadata
XML of that image (--ida-metadata; the "Metadata" download of the collection holds one XML per image)
and, for images without an XML, from the ADNIMERGE2 MRIQC table (--adnimerge2, screening MPRAGE series,
most frequent value per participant). The column scanner_source records which one supplied them. The
XML also gives the field strength and the in-plane pixel spacing, kept for information.

ADNI data cannot be redistributed: write the output next to the ADNI files, never into the repository.

Usage: python scripts/adni_network_cohort.py --clinical clinical_metadata.csv --folds results/adni/adni_folds.csv
           --manifest manifest_images.csv --adni-root "D:/ADNI1_Complete 1Yr 1.5T/ADNI"
           [--ida-metadata "D:/ADNI1_Complete_1Yr_1.5T_IDA_Metadata/ADNI" ...]
           [--adnimerge2 path/to/ADNIMERGE2] --out adni_network_cohort.csv
"""
from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

MANUFACTURERS = ("SIEMENS", "GE", "PHILIPS")


def normalise_manufacturer(value) -> object:
    """'GE MEDICAL SYSTEMS' -> 'GE', 'Philips Medical Systems' -> 'PHILIPS', 'SIEMENS' -> 'SIEMENS'."""
    if value is None or pd.isna(value):
        return pd.NA
    text = str(value).upper()
    for name in MANUFACTURERS:
        if re.search(rf"(^|[^A-Z]){name}([^A-Z]|$)", text):
            return name
    return text


def read_ida_metadata(dirs: list[Path]) -> pd.DataFrame:
    """One row per IDA metadata XML: image_id (the processed image UID in the file name, ..._I118671.xml),
    manufacturer, scanner_model, field_strength_t, pixel_spacing_mm."""
    rows = []
    for d in dirs:
        for f in sorted(Path(d).glob("*.xml")):
            root = ET.parse(f).getroot()
            prot = {p.get("term"): (p.text or "").strip() for p in root.iter("protocol") if p.get("term")}
            m = re.search(r"_I(\d+)\.xml$", f.name)
            uids = [e.text.strip() for e in root.iter("imageUID") if e.text]
            image_id = m.group(1) if m else (uids[0] if uids else None)
            rows.append({"image_id": image_id, "manufacturer": normalise_manufacturer(prot.get("Manufacturer")),
                         "scanner_model": prot.get("Mfg Model") or pd.NA,
                         "field_strength_t": pd.to_numeric(prot.get("Field Strength"), errors="coerce"),
                         "pixel_spacing_mm": pd.to_numeric(prot.get("Pixel Spacing X"), errors="coerce")})
    meta = pd.DataFrame(rows).dropna(subset=["image_id"]).drop_duplicates("image_id")
    meta["image_id"] = meta["image_id"].astype(str)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clinical", type=Path, required=True)
    ap.add_argument("--folds", type=Path, required=True, help="adni_folds.csv written by adni_replication.py")
    ap.add_argument("--manifest", type=Path, required=True, help="manifest_images.csv of the longitudinal project")
    ap.add_argument("--adni-root", type=Path, required=True, help="the ADNI/ folder of the IDA download")
    ap.add_argument("--ida-metadata", type=Path, nargs="*", default=[],
                    help="folder(s) of IDA metadata XMLs, one per image (the collection's Metadata download)")
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

    for col in ("manufacturer", "scanner_model", "field_strength_t", "pixel_spacing_mm", "scanner_source"):
        cohort[col] = pd.NA

    if args.ida_metadata:
        # The XML of the baseline image itself: exact, one per image.
        meta = read_ida_metadata(args.ida_metadata)
        key = cohort["image_id"].astype(str).str.replace(r"^I", "", regex=True)
        hit = meta.set_index("image_id").reindex(key)
        found = hit["manufacturer"].notna().to_numpy()
        for col in ("manufacturer", "scanner_model", "field_strength_t", "pixel_spacing_mm"):
            cohort.loc[found, col] = hit[col].to_numpy()[found]
        cohort.loc[found, "scanner_source"] = "ida_xml"
        print(f"IDA metadata: {len(meta)} XMLs read, {int(found.sum())} of {len(cohort)} baseline images matched")

    if args.adnimerge2 is not None and cohort["manufacturer"].isna().any():
        # Fallback: MRIQC lists the raw series (not the processed image IDs) with the scanner; take the screening
        # MPRAGE series of each participant and the most frequent manufacturer and model among them.
        import pyreadr  # noqa: E402 - optional dependency
        qc = list(pyreadr.read_r(str(args.adnimerge2 / "data" / "MRIQC.rda")).values())[0]
        qc["ParticipantID"] = qc["ParticipantID"].astype(str)
        sc = qc[qc["VISCODE2"].astype(str).isin(["sc", "scmri", "bl"])
                & qc["SeriesDescription"].astype(str).str.upper().str.contains("MP-?RAGE", regex=True)]

        def mode_or_na(s):
            s = s.dropna().astype(str).str.upper()
            return s.mode().iloc[0] if len(s) else pd.NA

        mode = sc.groupby("ParticipantID")[["ScannerManufacturer", "ScannerModel"]].agg(mode_or_na)
        missing = cohort["manufacturer"].isna().to_numpy()
        fill = mode.reindex(cohort.loc[missing, "subject_id"])
        cohort.loc[missing, "manufacturer"] = fill["ScannerManufacturer"].map(normalise_manufacturer).to_numpy()
        cohort.loc[missing, "scanner_model"] = fill["ScannerModel"].to_numpy()
        cohort.loc[missing & cohort["manufacturer"].notna().to_numpy(), "scanner_source"] = "mriqc"
        print(f"MRIQC fallback used for {int((cohort['scanner_source'] == 'mriqc').sum())} participants")

    print("manufacturer:", cohort["manufacturer"].value_counts(dropna=False).to_dict())
    cohort = cohort[["subject_id", "site", "fold", "label", "cdglobal", "mmse", "age", "sex",
                     "image_id", "description", "image_relpath", "manufacturer", "scanner_model",
                     "field_strength_t", "pixel_spacing_mm", "scanner_source"]]
    cohort.to_csv(args.out, index=False)
    print(f"{len(cohort)} participants, {int((cohort['label'] == 'Demented').sum())} Demented, "
          f"folds {sorted(cohort['fold'].unique())}; written to {args.out}")


if __name__ == "__main__":
    main()
