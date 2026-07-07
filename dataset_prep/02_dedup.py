#!/usr/bin/env python3
"""Step 1.2 — hash-based de-duplication of the merged raw dataset (01 output).

Runs BEFORE orientation / landmarks so no wasted GPU work is spent on duplicates.

Two scans are duplicates when the SHA-256 of their (lower.stl, upper.stl) pair is
identical. On a collision the LOWEST id is kept — since the original Bits2Bites
(A) patients occupy ids 1..N, this always keeps the published A patient and drops
the Bits2Bites2 (B) copy. Among B-only collisions the earlier-numbered one wins.

Kept patients are renumbered to a contiguous 1..K range (A ids 1..N are the
lowest and therefore unchanged; only B ids shift down to close the gaps left by
removed duplicates). Annotations.csv, per-patient folders, photos/<id>/ and any
landmarks/dental_<id:04d>.json are all renumbered together so ids stay consistent.

Everything removed is logged to reports/dedup_removed.csv.

Operates in place on --dataset by default; pass --output to write a de-duplicated
copy instead and leave the input untouched.

Usage:
    python 02_dedup.py --dataset /work/grana_maxillo/lborghi/datasets/Bits2Bites_merged
    # or, non-destructive:
    python 02_dedup.py --dataset <merged> --output <merged_dedup>
"""
import argparse
import csv
import hashlib
import shutil
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, required=True, help="Merged raw dataset from 01_merge_convert.py")
    p.add_argument("--output", type=Path, default=None, help="Write de-duplicated copy here (default: edit --dataset in place)")
    p.add_argument("--force", action="store_true", help="Overwrite --output if it exists")
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_annotations(dataset: Path):
    with open(dataset / "Annotations.csv", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [r for r in reader if r]
    return header, rows


def write_csv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\r\n")
        w.writerow(header)
        w.writerows(rows)


def read_id_mapping(dataset: Path):
    """Return {id: (source, original_folder)} from step 01's mapping (may be empty)."""
    path = dataset / "reports" / "id_mapping.csv"
    if not path.is_file():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {r["new_id"]: (r.get("source", ""), r["original_folder"]) for r in reader}


def main():
    args = parse_args()
    src = args.dataset
    header, rows = read_annotations(src)
    id_map = read_id_mapping(src)

    ids = sorted((int(r[0]) for r in rows))
    row_by_id = {int(r[0]): r for r in rows}

    # First occurrence of each hash wins; later ones are duplicates to drop.
    seen = {}          # hash -> kept id
    keep = []          # ids kept, ascending
    removed = []       # [removed_id, original_folder, kept_id, hash]
    for pid in ids:
        folder = src / str(pid)
        digest = hashlib.sha256()
        for jaw in ("lower", "upper"):
            digest.update(sha256_file(folder / f"{jaw}.stl").encode())
        h = digest.hexdigest()
        if h in seen:
            removed.append([pid, id_map.get(str(pid), ("", ""))[1], seen[h], h])
        else:
            seen[h] = pid
            keep.append(pid)

    # Contiguous remap: kept ids (ascending) -> 1..K.
    remap = {old: new for new, old in enumerate(keep, start=1)}

    dst = args.output if args.output is not None else src
    if args.output is not None:
        if dst.exists():
            if not args.force:
                print(f"error: {dst} already exists (pass --force)", file=sys.stderr)
                sys.exit(1)
            shutil.rmtree(dst)
        dst.mkdir(parents=True)

    _apply(src, dst, header, row_by_id, id_map, keep, remap, removed, in_place=args.output is None)

    print(f"Kept {len(keep)} / {len(ids)} patients; removed {len(removed)} duplicates.")
    print(f"Report -> {dst / 'reports' / 'dedup_removed.csv'}")


def _apply(src, dst, header, row_by_id, id_map, keep, remap, removed, in_place):
    """Materialize the remap either in place (src==dst) or into a fresh copy."""
    if in_place:
        # Drop duplicate folders/photos first.
        removed_ids = {r[0] for r in removed}
        for pid in removed_ids:
            shutil.rmtree(src / str(pid), ignore_errors=True)
            shutil.rmtree(src / "photos" / str(pid), ignore_errors=True)
        # Renumber via a temp suffix to avoid collisions during renaming.
        _renumber_in_place(src, keep, remap)
    else:
        for old in keep:
            new = remap[old]
            shutil.copytree(src / str(old), dst / str(new))
            photo = src / "photos" / str(old)
            if photo.is_dir():
                shutil.copytree(photo, dst / "photos" / str(new))
            lm = src / "landmarks" / f"dental_{old:04d}.json"
            if lm.is_file():
                (dst / "landmarks").mkdir(parents=True, exist_ok=True)
                shutil.copy2(lm, dst / "landmarks" / f"dental_{new:04d}.json")

    new_rows = []
    for old in keep:
        row = list(row_by_id[old])
        row[0] = str(remap[old])
        new_rows.append(row)
    write_csv(dst / "Annotations.csv", header, new_rows)

    new_mapping = [[str(remap[old]), *id_map.get(str(old), ("", str(old)))] for old in keep]
    write_csv(dst / "reports" / "id_mapping.csv", ["new_id", "source", "original_folder"], new_mapping)

    # Preserve the drop report from step 01, append dedup removals separately.
    if in_place:
        # dropped_no_manual.csv already lives under src/reports.
        pass
    else:
        old_drop = src / "reports" / "dropped_no_manual.csv"
        if old_drop.is_file():
            shutil.copy2(old_drop, dst / "reports" / "dropped_no_manual.csv")

    write_csv(dst / "reports" / "dedup_removed.csv",
              ["removed_id", "original_folder", "kept_id", "sha256"], removed)


def _renumber_in_place(root: Path, keep, remap):
    """Rename kept ids to their new numbers using a two-phase temp rename."""
    # Phase 1: move everything to a temp name keyed by new id.
    for old in keep:
        new = remap[old]
        if old == new:
            continue
        (root / str(old)).rename(root / f"__tmp_{new}")
        photo = root / "photos" / str(old)
        if photo.is_dir():
            photo.rename(root / "photos" / f"__tmp_{new}")
        lm = root / "landmarks" / f"dental_{old:04d}.json"
        if lm.is_file():
            lm.rename(root / "landmarks" / f"__tmp_{new}.json")
    # Phase 2: strip the temp prefix.
    for old in keep:
        new = remap[old]
        if old == new:
            continue
        (root / f"__tmp_{new}").rename(root / str(new))
        tphoto = root / "photos" / f"__tmp_{new}"
        if tphoto.is_dir():
            tphoto.rename(root / "photos" / str(new))
        tlm = root / "landmarks" / f"__tmp_{new}.json"
        if tlm.is_file():
            tlm.rename(root / "landmarks" / f"dental_{new:04d}.json")


if __name__ == "__main__":
    main()
