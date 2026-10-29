"""Build the baseline volume table for the ADNI replication from the ADNIMERGE2 R data package.

Reads ADSP_PHC_T1_FS.rda (ADSP Phenotype Harmonization Consortium composite FreeSurfer scores, raw metrics)
and writes one row per study subject with the same feature definitions as on OASIS-1:
  Hippocampus = Left_Hippocampus + Right_Hippocampus (mm^3, FreeSurfer aseg)
  WholeBrain  = BrainSegVol;  ICV = EstimatedTotalIntraCranialVol  (so that WholeBrain/ICV = BrainSegVol/ICV)
The baseline scan is the screening or baseline visit (VISCODE2 sc, scmri or bl); when a subject has both a
1.5 T and a 3 T baseline scan, the 1.5 T scan of the ADNI1 main study is kept. The output has the column
layout that scripts/adni_replication.py expects from ADNIMERGE (PTID, VISCODE, Hippocampus, WholeBrain, ICV,
FSVERSION). ADNI data cannot be redistributed; the output stays outside the repository.

Usage: python scripts/adni_build_volumes.py --package path/to/ADNIMERGE2 --clinical clinical_metadata.csv --out adni_volumes_bl.csv
"""
import argparse
from pathlib import Path

import pandas as pd
import pyreadr

BASELINE_VISITS = ("sc", "scmri", "bl")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--package", type=Path, required=True, help="root of the extracted ADNIMERGE2 R package")
    ap.add_argument("--clinical", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    fs = pyreadr.read_r(str(args.package / "data" / "ADSP_PHC_T1_FS.rda"))["ADSP_PHC_T1_FS"]
    clinical = pd.read_csv(args.clinical)
    subjects = set(clinical.loc[clinical["timepoint"] == "bl", "subject_id"].astype(str))

    base = fs[fs["PTID"].isin(subjects) & fs["VISCODE2"].astype(str).isin(BASELINE_VISITS)].copy()
    base["FieldSgth"] = pd.to_numeric(base["FieldSgth"], errors="coerce")
    base = base.sort_values(["PTID", "FieldSgth", "PHC_SCANDATE"]).drop_duplicates("PTID", keep="first")
    out = pd.DataFrame({
        "PTID": base["PTID"].to_numpy(), "VISCODE": "bl",
        "Hippocampus": (base["Left_Hippocampus"] + base["Right_Hippocampus"]).to_numpy(),
        "WholeBrain": base["BrainSegVol"].to_numpy(), "ICV": base["EstimatedTotalIntraCranialVol"].to_numpy(),
        "FieldStrength": base["FieldSgth"].to_numpy(), "ScanAge": base["PHC_Age_T1"].to_numpy(),
        "FSVERSION": "ADSP PHC composite FreeSurfer, raw metrics",
    })
    out.to_csv(args.out, index=False)
    print(f"{len(out)}/{len(subjects)} study subjects with a baseline FreeSurfer row; "
          f"field strength {out['FieldStrength'].value_counts().to_dict()}; written to {args.out}")
    print(out[["Hippocampus", "WholeBrain", "ICV"]].describe().round(0).to_string())


if __name__ == "__main__":
    main()
