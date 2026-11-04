"""Build the cohort file for the age-matched training run on OASIS-1 (Q4 of docs/prespecification_networks_adni.md):
the subjects aged 60 or over of the shared split, with their labels and fold numbers, in the format that
scripts/slice_networks.py reads (subject_id, label, fold).

The file carries labels, so it stays outside the repository (OASIS Data Use Agreement); upload it to Drive as
alzheimer_anfis_outputs/results/oasis60_cohort.csv and set COHORT = 'oasis60' in
notebooks/slice_networks_multiseed_colab.ipynb.

Usage: python scripts/oasis60_cohort.py --cohort path/to/oasis_master_features_labels.csv --out oasis60_cohort.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as ev  # noqa: E402

MIN_AGE = 60


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cohort", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=ev.REPO / "data")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cohort, y, folds, _, _, _ = ev.load(args.cohort, args.data)
    old = cohort["age"].to_numpy(dtype=float) >= MIN_AGE
    out = pd.DataFrame({"subject_id": cohort["subject_id"][old].astype(str),
                        "label": cohort["label"][old], "fold": folds[old]})
    assert set(out["label"]) <= {"Demented", "Nondemented"}
    out.to_csv(args.out, index=False)
    print(f"{len(out)} subjects aged {MIN_AGE} or over ({int((out['label'] == 'Demented').sum())} demented); "
          f"test folds {out['fold'].value_counts().sort_index().tolist()}; written to {args.out}")


if __name__ == "__main__":
    main()
