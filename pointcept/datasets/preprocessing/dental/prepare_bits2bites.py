"""Prepare the raw Bits2Bites dataset for `DentalDataset`.

Builds a `"Mesh"` point cloud per patient from the raw STL scans, optionally
merging in per-tooth landmark JSONs
(`datasets/Bits2Bites/landmarks/dental_<id>.json`, not part of the public
release — see `BITS2BITES.md`) if present, translates `Annotations.csv` into
the Italian label vocabulary `pointcept/datasets/dental.py` expects, and
stratifies the result into folds via `create_balanced_folds` (same routine
`preprocess_dentalnet.py` uses). See `BITS2BITES.md` for the full
reproduction workflow.
"""

import argparse
import os
import json
import shutil
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from preprocess_dentalnet import create_balanced_folds, load_points_from_stl

RIGHT_LEFT_MAP = {
    "Class I": "prima classe",
    "Class II Edge to Edge": "seconda classe",
    "Class II Full": "seconda classe",
    "Class III": "terza classe",
    "Unknown": "non valutabile",
}
ANTERIOR_MAP = {
    "Normal": "normale",
    "Deep Bite": "profondo",
    "Open Bite": "aperto",
    "Inverted Bite": "inverso",
    # "Unknown" -> a token outside dental.py's vocab, so the head is ignored (-1).
    # A's Anterior column never has Unknown; Bits2Bites2 can (unevaluable axis).
    "Unknown": "non valutabile",
}
MIDLINE_MAP = {
    "Centered": "centrata",
    "Deviated": "deviata",
    "Unknown": "non valutabile",
}


def map_transverse(value: str) -> str:
    """`Normal` / `Cross ...` / `Scissor ...` / compound `Cross ... / Scissor ...` -> vocab.

    `Unknown` (Bits2Bites2 unevaluable axis) -> out-of-vocab token, so the head is
    ignored (-1), matching how sagittal Unknown is already handled.
    """
    value = value.strip()
    if value == "Normal":
        return "normale"
    if value == "Unknown":
        return "non valutabile"
    if "Cross" in value:
        return "cross"
    if "Scissor" in value:
        return "scissor"
    raise ValueError(f"Unrecognized Transversal Bite value: {value!r}")


def build_labels_csv(dataset_root: Path, output_csv: Path) -> None:
    df = pd.read_csv(dataset_root / "Annotations.csv")
    out = pd.DataFrame(
        {
            "PAZIENTE": df["Patient"],
            "CLASSE DX": df["Right Class"].map(RIGHT_LEFT_MAP),
            "CLASSE SX": df["Left Class"].map(RIGHT_LEFT_MAP),
            "MORSO ANTERIORE": df["Anterior Bite"].map(ANTERIOR_MAP),
            "TRASVERSALE (senza id denti)": df["Transversal Bite"].map(map_transverse),
            "LINEE MEDIANE": df["Median Lines"].map(MIDLINE_MAP),
        }
    )
    if out.isna().any().any():
        bad = df[out.isna().any(axis=1)]
        raise ValueError(f"Unmapped label values for patients: {bad['Patient'].tolist()}")
    out.to_csv(output_csv, index=False)


def patient_ids_from_annotations(dataset_root: Path) -> list[int]:
    df = pd.read_csv(dataset_root / "Annotations.csv")
    return [int(patient_id) for patient_id in df["Patient"].tolist()]


def merge_patient(patient_id: int, dataset_root: Path, merged_dir: Path) -> None:
    """Merge per-tooth landmarks (if available) with mesh points into one sample.

    The public Bits2Bites release ships only meshes + Annotations.csv — no
    landmarks. If `datasets/Bits2Bites/landmarks/dental_<id>.json` isn't
    present, this falls back to a mesh-only sample (Mesh points only, no
    per-point landmark one-hot bits set) instead of erroring.
    """
    landmarks_path = dataset_root / "landmarks" / f"dental_{patient_id:04d}.json"
    if landmarks_path.is_file():
        with open(landmarks_path) as f:
            merged = json.load(f)
        description = "landmarks+mesh"
    else:
        merged = {"version": "1.1", "key": f"dental_{patient_id:04d}", "objects": []}
        description = "mesh"

    for jaw in ("lower", "upper"):
        stl_path = dataset_root / str(patient_id) / f"{jaw}.stl"
        vertices = load_points_from_stl(str(stl_path))
        for vertex in vertices:
            merged["objects"].append(
                {"key": f"uuid_{uuid.uuid4().int}", "class": "Mesh", "coord": vertex.tolist()}
            )

    merged["description"] = description
    with open(merged_dir / f"dental_{patient_id:04d}.json", "w") as f:
        json.dump(merged, f)


def main():
    parser = argparse.ArgumentParser(description="Prepare datasets/Bits2Bites for DentalDataset")
    parser.add_argument(
        "--dataset-root",
        default="../../datasets/Bits2Bites",
        help="Path to the raw Bits2Bites download (STLs + Annotations.csv + landmarks/)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/dental_landmarks_mesh",
        help="Destination data_root matching configs/dental/_base_dental.py",
    )
    parser.add_argument("--num-patients", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, max(1, os.cpu_count() or 1)),
        help="Number of parallel workers for mesh/landmark merging",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    merged_dir = output_dir / "_merged"

    output_dir.mkdir(parents=True, exist_ok=True)
    if merged_dir.exists():
        shutil.rmtree(merged_dir)
    merged_dir.mkdir(parents=True)

    has_landmarks = (dataset_root / "landmarks").is_dir()
    print(
        f"Landmarks {'found' if has_landmarks else 'NOT found'} at "
        f"{dataset_root / 'landmarks'} -> building "
        f"{'landmarks+mesh' if has_landmarks else 'mesh-only'} samples."
    )

    print("Translating Annotations.csv to labels.csv ...")
    labels_csv = output_dir / "labels.csv"
    build_labels_csv(dataset_root, labels_csv)

    patient_ids = patient_ids_from_annotations(dataset_root)
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    total = len(patient_ids)
    if args.workers == 1:
        for idx, patient_id in enumerate(patient_ids, 1):
            print(f"Merging patient {idx}/{total} (id {patient_id})")
            merge_patient(patient_id, dataset_root, merged_dir)
    else:
        print(f"Merging {total} patients with {args.workers} workers ...")
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(merge_patient, patient_id, dataset_root, merged_dir): patient_id
                for patient_id in patient_ids
            }
            for idx, future in enumerate(as_completed(futures), 1):
                patient_id = futures[future]
                future.result()
                print(f"Merged patient {idx}/{total} (id {patient_id})")

    print("Stratifying into folds ...")
    create_balanced_folds(merged_dir, labels_csv, output_dir, n_folds=5)
    shutil.rmtree(merged_dir)
    print(f"Done. Data ready at {output_dir}")


if __name__ == "__main__":
    main()
