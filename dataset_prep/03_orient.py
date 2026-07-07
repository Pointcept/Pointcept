#!/usr/bin/env python3
"""Step 2 — orient every scan into one canonical frame with the ios_orienter
PointNet regressor, keeping upper and lower locked together.

ios_orienter predicts a restoring rotation R from an area-weighted surface
sample of the scan (iterative refinement + test-time augmentation averaged in
SO(3)); `canonical ~= R @ (x - centroid)`. We predict it ONCE from the MERGED
upper+lower point cloud (its most accurate mode, ~4.2 deg mean error) so both
jaws get the identical rotation and their bite relationship is preserved. We
apply ONLY that rotation, about the merged centroid, keeping the ORIGINAL mm
dimensions (no unit-sphere rescale) and original position:
    oriented = (verts - center) @ matrix + center      (matrix = R.T)
Running it independently per jaw would send each jaw to its own frame and
destroy the bite relationship.

The whole merged set (A + B) is oriented so every sample shares one frame.

Outputs (a raw Bits2Bites-format dataset, ready for prepare_bits2bites.py):
    <out>/Annotations.csv                 (copied)
    <out>/<id>/{lower,upper}.stl          (oriented, faces preserved)
    <out>/landmarks/dental_<id:04d>.json  (A landmarks carried through the SAME
                                           transform; B gets its own in step 3)
    <out>/photos/<id>/...                 (copied, unchanged)
    <out>/transforms/<id>.json            (center, matrix, confidence_deg — invertible)
    <out>/qa/overview_before_after.png    (montage, --qa N patients)
    <out>/qa/<id>_before_after.png        (per scan, with --qa-per-scan)

GPU strongly recommended (surface sampling + net + TTA). Run via
sbatch_orient.sh on a GPU node.

Usage:
    python 03_orient.py \
        --dataset  /work/grana_maxillo/lborghi/datasets/Bits2Bites_merged \
        --output   /work/grana_maxillo/lborghi/datasets/Bits2Bites_oriented \
        --checkpoint dataset_prep/weights/orient_pointnet_best.pt \
        --qa 10
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh

IOS_SRC = Path(__file__).resolve().parent / "ios_orienter"
sys.path.insert(0, str(IOS_SRC))
from infer import load_model, predict_rotation  # noqa: E402
from stl_io import read_stl, sample_surface  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, required=True, help="Merged/de-duplicated raw dataset")
    p.add_argument("--output", type=Path, required=True, help="Where to write the oriented dataset")
    p.add_argument("--checkpoint", type=Path, required=True, help="ios_orienter PointNet model .pt")
    p.add_argument("--sample-points", type=int, default=30000, help="area-weighted surface points sampled per patient before TTA")
    p.add_argument("--tta", type=int, default=8, help="test-time augmentation rotations averaged per refinement step")
    p.add_argument("--iters", type=int, default=3, help="predict/apply/re-predict refinement iterations")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for surface sampling + TTA")
    p.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    p.add_argument("--qa", type=int, default=10, help="patients in the overview QA montage (0 disables)")
    p.add_argument("--qa-per-scan", action="store_true", help="also write one before/after PNG per patient")
    p.add_argument("--qa-points", type=int, default=4000, help="points sub-sampled per jaw in QA plots")
    p.add_argument("--limit", type=int, default=None, help="process only the first N patients (debug)")
    p.add_argument("--force", action="store_true", help="reprocess a patient even if its outputs already exist")
    return p.parse_args()


def load_mesh(path: Path):
    return trimesh.load(path, force="mesh", process=False)


def predict_transform(lower_path: Path, upper_path: Path, model, cfg, device, args, seed):
    """Return (center, matrix, confidence_deg) from the merged upper+lower surface.

    Area-weighted surface sampling (not vertex sampling) — this is what
    ios_orienter was trained on. `matrix` composes with apply_transform's
    `(verts - center) @ matrix` convention (matrix = R.T, R the restoring
    rotation such that canonical ~= R @ (x - centroid)).
    """
    rng = np.random.default_rng(seed)
    all_pts, all_nrm = [], []
    for p in (lower_path, upper_path):
        tri, nrm = read_stl(str(p))
        s_pts, s_nrm = sample_surface(tri, nrm, args.sample_points, rng)
        all_pts.append(s_pts)
        all_nrm.append(s_nrm)
    pts = np.concatenate(all_pts, axis=0)
    nrms = np.concatenate(all_nrm, axis=0)
    centroid = pts.mean(axis=0)

    r, agreement = predict_rotation(
        model, cfg, pts - centroid, nrms, device,
        n_iters=args.iters, tta=args.tta, seed=seed,
    )
    center = torch.from_numpy(centroid.astype(np.float32))
    matrix = torch.from_numpy(r.T.astype(np.float32))
    return center, matrix, agreement


def apply_transform(verts: torch.Tensor, center, matrix) -> torch.Tensor:
    """Rotate about the centroid, preserving original mm scale and position."""
    return (verts - center) @ matrix + center


def orient_patient(pid: int, dataset: Path, out: Path, model, cfg, device, args):
    src = dataset / str(pid)
    lower_path, upper_path = src / "lower.stl", src / "upper.stl"
    lower_mesh = load_mesh(lower_path)
    upper_mesh = load_mesh(upper_path)
    lower_v = torch.from_numpy(np.asarray(lower_mesh.vertices, dtype=np.float32))
    upper_v = torch.from_numpy(np.asarray(upper_mesh.vertices, dtype=np.float32))

    center, matrix, confidence_deg = predict_transform(
        lower_path, upper_path, model, cfg, device, args, seed=args.seed
    )

    oriented_lower = apply_transform(lower_v, center, matrix)
    oriented_upper = apply_transform(upper_v, center, matrix)

    dst = out / str(pid)
    dst.mkdir(parents=True, exist_ok=True)
    lower_mesh.vertices = oriented_lower.numpy()
    upper_mesh.vertices = oriented_upper.numpy()
    lower_mesh.export(dst / "lower.stl")
    upper_mesh.export(dst / "upper.stl")

    # Save the transform (reproducible / invertible).
    (out / "transforms").mkdir(parents=True, exist_ok=True)
    # Applied map is: oriented = (v - center) @ matrix + center  (rotation only,
    # original mm scale). Invert with: v = (oriented - center) @ matrix.T + center.
    (out / "transforms" / f"{pid}.json").write_text(json.dumps({
        "center": center.tolist(),
        "matrix": matrix.tolist(),
        "space": "original_mm",
        "confidence_deg": confidence_deg,
    }))

    # Carry A's landmarks through the SAME transform, if present.
    lm_src = dataset / "landmarks" / f"dental_{pid:04d}.json"
    if lm_src.is_file():
        lm = json.loads(lm_src.read_text())
        for obj in lm.get("objects", []):
            v = torch.tensor(obj["coord"], dtype=torch.float32).unsqueeze(0)
            obj["coord"] = apply_transform(v, center, matrix)[0].tolist()
        (out / "landmarks").mkdir(parents=True, exist_ok=True)
        (out / "landmarks" / f"dental_{pid:04d}.json").write_text(json.dumps(lm))

    return {
        "before": {"lower": lower_v.numpy(), "upper": upper_v.numpy()},
        "after": {"lower": oriented_lower.numpy(), "upper": oriented_upper.numpy()},
    }


# ----------------------------- QA rendering --------------------------------- #

def _subsample(arr, n):
    if len(arr) <= n:
        return arr
    idx = np.random.default_rng(0).choice(len(arr), n, replace=False)
    return arr[idx]


def _draw_cell(ax, lower, upper, n, title):
    import matplotlib.pyplot as plt  # noqa: F401
    lo, up = _subsample(lower, n), _subsample(upper, n)
    ax.scatter(lo[:, 0], lo[:, 1], lo[:, 2], s=0.5, c="tab:blue", label="lower")
    ax.scatter(up[:, 0], up[:, 1], up[:, 2], s=0.5, c="tab:red", label="upper")
    ax.set_title(title, fontsize=8)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


def render_overview(records, path: Path, n_points):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = len(records)
    fig = plt.figure(figsize=(6, 3 * rows))
    for i, (pid, rec) in enumerate(records):
        ax_b = fig.add_subplot(rows, 2, 2 * i + 1, projection="3d")
        _draw_cell(ax_b, rec["before"]["lower"], rec["before"]["upper"], n_points, f"#{pid} before")
        ax_a = fig.add_subplot(rows, 2, 2 * i + 2, projection="3d")
        _draw_cell(ax_a, rec["after"]["lower"], rec["after"]["upper"], n_points, f"#{pid} after")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def render_per_scan(pid, rec, path: Path, n_points):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8, 4))
    ax_b = fig.add_subplot(1, 2, 1, projection="3d")
    _draw_cell(ax_b, rec["before"]["lower"], rec["before"]["upper"], n_points, f"#{pid} before")
    ax_a = fig.add_subplot(1, 2, 2, projection="3d")
    _draw_cell(ax_a, rec["after"]["lower"], rec["after"]["upper"], n_points, f"#{pid} after")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --------------------------------- main ------------------------------------- #

def patient_ids(dataset: Path):
    import csv
    with open(dataset / "Annotations.csv", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        return sorted(int(r[0]) for r in reader if r)


def main():
    args = parse_args()
    import shutil

    args.output.mkdir(parents=True, exist_ok=True)

    shutil.copy2(args.dataset / "Annotations.csv", args.output / "Annotations.csv")
    for extra in ("photos", "reports"):
        if (args.dataset / extra).is_dir():
            shutil.copytree(args.dataset / extra, args.output / extra, dirs_exist_ok=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, cfg = load_model(str(args.checkpoint), device)
    print(f"Loaded ios_orienter model on {device}, points={cfg['points']}, in_ch={cfg['in_ch']}")

    ids = patient_ids(args.dataset)
    if args.limit:
        ids = ids[: args.limit]

    qa_records = []
    processed, skipped = 0, 0
    for k, pid in enumerate(ids):
        dst = args.output / str(pid)
        done = (dst / "lower.stl").is_file() and (dst / "upper.stl").is_file()
        if done and not args.force:
            skipped += 1
            continue
        rec = orient_patient(pid, args.dataset, args.output, model, cfg, device, args)
        processed += 1
        keep_qa = (args.qa and len(qa_records) < args.qa) or args.qa_per_scan
        if keep_qa:
            if args.qa_per_scan:
                render_per_scan(pid, rec, args.output / "qa" / f"{pid}_before_after.png", args.qa_points)
            if args.qa and len(qa_records) < args.qa:
                qa_records.append((pid, rec))
        if (k + 1) % 25 == 0:
            print(f"  oriented {k + 1}/{len(ids)}")

    if qa_records:
        render_overview(qa_records, args.output / "qa" / "overview_before_after.png", args.qa_points)

    print(f"Oriented {processed} patients ({skipped} already done, skipped) -> {args.output}")
    if args.qa:
        print(f"QA montage -> {args.output / 'qa' / 'overview_before_after.png'}")


if __name__ == "__main__":
    main()
