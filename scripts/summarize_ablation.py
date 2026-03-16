#!/usr/bin/env python3
import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


BEST_RE = re.compile(r"Best validation mIoU updated to:\s*([0-9.]+)")
VAL_RE = re.compile(r"Val result: mIoU/mAcc/allAcc\s*([0-9.]+)/([0-9.]+)/([0-9.]+)")
TEST_LINE_RE = re.compile(r"test\.py line 340")
CLASS_RE = re.compile(
    r"Class_(\d+)\s*-\s*([A-Za-z0-9_]+)\s*Result:\s*iou/accuracy\s*([0-9.]+)/([0-9.]+)"
)
SEED_DIR_RE = re.compile(r"^(.*)_seed(\d+)$")


@dataclass
class Summary:
    exp_dir: str
    train_log: str
    best_miou: Optional[float]
    final_miou: Optional[float]
    final_macc: Optional[float]
    final_allacc: Optional[float]
    final_cls_iou: Dict[str, float]


def _read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().splitlines()


def _parse_train_log(train_log: str) -> Summary:
    lines = _read_lines(train_log)
    best_vals: List[float] = []
    final_metrics: Optional[Tuple[float, float, float]] = None
    final_cls_iou: Dict[str, float] = {}

    # best mIoU during training
    for line in lines:
        m = BEST_RE.search(line)
        if m:
            best_vals.append(float(m.group(1)))

    # final precise eval (printed by tester)
    # Strategy: find last "test.py line 340" that contains val result.
    for i in range(len(lines) - 1, -1, -1):
        if not TEST_LINE_RE.search(lines[i]):
            continue
        m = VAL_RE.search(lines[i])
        if m:
            final_metrics = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
            # parse following per-class lines (usually next few lines)
            for j in range(i + 1, min(i + 32, len(lines))):
                cm = CLASS_RE.search(lines[j])
                if not cm:
                    continue
                cls_name = cm.group(2)
                iou = float(cm.group(3))
                final_cls_iou[cls_name] = iou
            break

    best_miou = max(best_vals) if best_vals else None
    final_miou = final_metrics[0] if final_metrics else None
    final_macc = final_metrics[1] if final_metrics else None
    final_allacc = final_metrics[2] if final_metrics else None
    exp_dir = os.path.dirname(train_log)
    return Summary(
        exp_dir=exp_dir,
        train_log=train_log,
        best_miou=best_miou,
        final_miou=final_miou,
        final_macc=final_macc,
        final_allacc=final_allacc,
        final_cls_iou=final_cls_iou,
    )


def _find_train_logs(root: str) -> List[str]:
    out: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        if "train.log" in filenames:
            out.append(os.path.join(dirpath, "train.log"))
    out.sort()
    return out


def _fmt(v: Optional[float]) -> str:
    return "-" if v is None else f"{v:.4f}"


def _mean_std(vals: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not vals:
        return None, None
    mean = sum(vals) / len(vals)
    if len(vals) < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return mean, var**0.5


def _group_name(exp_dir: str) -> str:
    base = os.path.basename(exp_dir.rstrip("/"))
    m = SEED_DIR_RE.match(base)
    if m:
        return m.group(1)
    return base


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Pointcept train logs.")
    parser.add_argument(
        "--root",
        default="exp/tomato_gs2pc",
        help="Root directory to search for exp/*/train.log",
    )
    parser.add_argument(
        "--group",
        action="store_true",
        help="Group runs by experiment name (e.g. *_seed0/_seed1) and report mean/std.",
    )
    args = parser.parse_args()

    train_logs = _find_train_logs(args.root)
    if not train_logs:
        raise SystemExit(f"No train.log found under: {args.root}")

    summaries = [_parse_train_log(p) for p in train_logs]

    if args.group:
        grouped: Dict[str, List[Summary]] = {}
        for s in summaries:
            grouped.setdefault(_group_name(s.exp_dir), []).append(s)

        rows: List[Tuple[str, int, Optional[float], Optional[float], Dict[str, Optional[float]], Dict[str, Optional[float]]]] = []
        for name, items in grouped.items():
            finals = [s.final_miou for s in items if s.final_miou is not None]
            mean_miou, std_miou = _mean_std([v for v in finals if v is not None])
            per_cls_vals: Dict[str, List[float]] = {}
            for s in items:
                for cls in ("background", "stem", "leaf", "flower"):
                    v = s.final_cls_iou.get(cls)
                    if v is None:
                        continue
                    per_cls_vals.setdefault(cls, []).append(v)
            per_cls_mean: Dict[str, Optional[float]] = {}
            per_cls_std: Dict[str, Optional[float]] = {}
            for cls in ("background", "stem", "leaf", "flower"):
                m, sd = _mean_std(per_cls_vals.get(cls, []))
                per_cls_mean[cls] = m
                per_cls_std[cls] = sd
            rows.append((name, len(items), mean_miou, std_miou, per_cls_mean, per_cls_std))

        rows.sort(key=lambda r: (-1 if r[2] is None else -r[2], r[0]))

        header = ["exp", "n", "final_mIoU_mean", "final_mIoU_std", "stem", "leaf", "flower"]
        print("\t".join(header))
        for name, n, mean_miou, std_miou, per_cls_mean, _per_cls_std in rows:
            print(
                "\t".join(
                    [
                        name,
                        str(n),
                        _fmt(mean_miou),
                        _fmt(std_miou),
                        _fmt(per_cls_mean.get("stem")),
                        _fmt(per_cls_mean.get("leaf")),
                        _fmt(per_cls_mean.get("flower")),
                    ]
                )
            )
        return

    # Sort by final mIoU (desc), then best mIoU (desc).
    summaries.sort(
        key=lambda s: (
            -1 if s.final_miou is None else -s.final_miou,
            -1 if s.best_miou is None else -s.best_miou,
        )
    )

    header = [
        "exp_dir",
        "best_mIoU",
        "final_mIoU",
        "final_mAcc",
        "final_allAcc",
        "bg",
        "stem",
        "leaf",
        "flower",
    ]
    print("\t".join(header))
    for s in summaries:
        row = [
            s.exp_dir,
            _fmt(s.best_miou),
            _fmt(s.final_miou),
            _fmt(s.final_macc),
            _fmt(s.final_allacc),
            _fmt(s.final_cls_iou.get("background")),
            _fmt(s.final_cls_iou.get("stem")),
            _fmt(s.final_cls_iou.get("leaf")),
            _fmt(s.final_cls_iou.get("flower")),
        ]
        print("\t".join(row))


if __name__ == "__main__":
    main()
