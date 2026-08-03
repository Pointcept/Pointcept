#!/usr/bin/env python3
import argparse
import logging
import os
import socket
import struct
import sys
import time
from dataclasses import dataclass, field
from types import ModuleType
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# --- MOCK POINTOPS (for environments without the extension) ---
if "pointops" not in sys.modules:
    mock = ModuleType("pointops")
    mock.query_and_group_count = lambda *args, **kwargs: None
    sys.modules["pointops"] = mock

try:
    from pointcept.engines.defaults import default_config_parser
    from pointcept.datasets.transform import Compose
    from pointcept.models.builder import build_model
except ImportError as exc:
    raise SystemExit(f"❌ Erreur Pointcept : {exc}") from exc


MAGIC_HEADER = b"\x28\x2a"
BYTES_PER_POINT = 12


@dataclass
class BufferState:
    points: List[Tuple[float, float, float]] = field(default_factory=list)
    last_update: float = field(default_factory=time.monotonic)
    last_frame_id: Optional[int] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Realtime 3DP classification over UDP with frame-safe buffering.",
    )
    parser.add_argument("--udp-ip", default="127.0.0.1")
    parser.add_argument("--udp-port", type=int, default=8888)
    parser.add_argument("--offset-id", type=int, default=11)
    parser.add_argument("--offset-count", type=int, default=15)
    parser.add_argument("--offset-data", type=int, default=17)
    parser.add_argument("--offset-frame-id", type=int, default=-1)
    parser.add_argument("--min-points", type=int, default=400)
    parser.add_argument("--max-points", type=int, default=1024)
    parser.add_argument("--sample-points", type=int, default=512)
    parser.add_argument("--frame-timeout", type=float, default=0.15)
    parser.add_argument("--height-corr-coeff", type=float, default=0.015)
    parser.add_argument(
        "--config",
        default=os.path.join(
            os.path.dirname(__file__),
            "..",
            "configs",
            "3dp",
            "cls-3dp-ptv3-v1m1-0-base.py",
        ),
        help="Path to Pointcept config (.py).",
    )
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(
            os.path.dirname(__file__),
            "..",
            "exp",
            "3dp",
            "cls-ptv3",
            "model",
            "model_last.pth",
        ),
        help="Path to checkpoint (.pth).",
    )
    parser.add_argument(
        "--class-names",
        nargs="+",
        default=["MARCHE", "ACCROUPI", "ESCALADE"],
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def setup_logger(debug: bool) -> logging.Logger:
    logger = logging.getLogger("realtime_3dp_cls_udp")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def resolve_path(path: str, repo_root: str, kind: str) -> str:
    if path and os.path.isfile(path):
        return path
    candidate = os.path.join(repo_root, path) if path else ""
    if candidate and os.path.isfile(candidate):
        return candidate
    raise FileNotFoundError(
        f"{kind} introuvable: '{path}'. "
        f"Essayez un chemin absolu ou relatif au dépôt: '{repo_root}'."
    )


def load_model(config_path: str, ckpt_path: str, logger: logging.Logger):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    config_path = resolve_path(config_path, repo_root, "Config")
    cfg = default_config_parser(config_path, options={})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg.model).eval().to(device)
    try:
        ckpt_path = resolve_path(ckpt_path, repo_root, "Checkpoint")
    except FileNotFoundError:
        ckpt_path = ""
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt.get("state_dict", ckpt.get("model", ckpt)), strict=False)
        logger.info("Checkpoint loaded from %s", ckpt_path)
    else:
        logger.warning("Checkpoint not found: %s", ckpt_path)
    infer_tf = Compose(cfg.infer.transform)
    return model, infer_tf, device


def extract_points(data: bytes, offset_data: int, count: int) -> List[Tuple[float, float, float]]:
    points = []
    for i in range(count):
        idx = offset_data + (i * BYTES_PER_POINT)
        if idx + BYTES_PER_POINT <= len(data):
            points.append(struct.unpack_from("<3f", data, idx))
    return points


def robust_height(points: np.ndarray) -> Tuple[float, float, float]:
    z_vals = points[:, 2]
    z_min = np.percentile(z_vals, 5)
    z_max = np.percentile(z_vals, 95)
    height = max(0.0, float(z_max - z_min))
    return float(z_min), float(z_max), height


def corrected_height(height: float, points: np.ndarray, coeff: float) -> float:
    centroid = np.mean(points[:, :2], axis=0)
    radial_dist = float(np.linalg.norm(centroid))
    return height * (1.0 + radial_dist * coeff)


def normalize_points(points: np.ndarray) -> np.ndarray:
    pts_norm = points.copy()
    centroid = np.mean(pts_norm, axis=0)
    pts_norm[:, :2] -= centroid[:2]
    pts_norm[:, 2] -= np.min(pts_norm[:, 2])
    return pts_norm


def build_model_input(
    points: np.ndarray, infer_tf: Compose, device: torch.device
) -> Dict[str, torch.Tensor]:
    data = {"coord": points[:, [0, 2, 1]]}
    processed = infer_tf(data)
    for key, value in processed.items():
        if isinstance(value, np.ndarray):
            processed[key] = torch.from_numpy(value).to(device)
        elif torch.is_tensor(value):
            processed[key] = value.to(device)
    coord = processed["coord"]
    processed["batch"] = torch.zeros(coord.shape[0], dtype=torch.long, device=device)
    processed["offset"] = torch.tensor([coord.shape[0]], dtype=torch.long, device=device)
    return processed


def log_histogram(logger: logging.Logger, title: str, values: np.ndarray) -> None:
    hist, bins = np.histogram(values, bins=8)
    bars = " ".join(f"{int(count):4d}" for count in hist)
    ranges = " ".join(f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(len(hist)))
    logger.debug("%s | bins: %s", title, ranges)
    logger.debug("%s | counts: %s", title, bars)


def maybe_sample(points: np.ndarray, sample_points: int) -> np.ndarray:
    if points.shape[0] <= sample_points:
        if points.shape[0] == sample_points:
            return points
        idx = np.random.choice(points.shape[0], sample_points, replace=True)
        return points[idx]
    idx = np.random.choice(points.shape[0], sample_points, replace=False)
    return points[idx]


def main() -> None:
    args = parse_args()
    logger = setup_logger(args.debug)

    logger.info("🚀 ANALYSEUR AVEC BUFFER FRAME-SAFE")
    logger.info("CTRL+C pour quitter")

    try:
        model, infer_tf, device = load_model(args.config, args.checkpoint, logger)
    except FileNotFoundError as exc:
        logger.error(
            "%s\nAstuce: lancez avec --config /chemin/vers/config.py",
            exc,
        )
        return
    logger.info("Modèle chargé sur %s", device)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.udp_ip, args.udp_port))
    sock.settimeout(0.5)

    buffers: Dict[int, BufferState] = {}

    try:
        while True:
            try:
                payload, _ = sock.recvfrom(4096)
            except socket.timeout:
                continue

            if not payload.startswith(MAGIC_HEADER):
                continue

            cid = struct.unpack_from("<H", payload, args.offset_id)[0]
            num_pts = struct.unpack_from("<H", payload, args.offset_count)[0]
            frame_id = None
            if args.offset_frame_id >= 0:
                frame_id = struct.unpack_from("<H", payload, args.offset_frame_id)[0]

            points = extract_points(payload, args.offset_data, num_pts)
            state = buffers.setdefault(cid, BufferState())
            now = time.monotonic()

            if frame_id is not None and state.last_frame_id is not None:
                if frame_id != state.last_frame_id:
                    logger.debug("Frame change for cid=%s: %s -> %s", cid, state.last_frame_id, frame_id)
                    state.points = []

            if now - state.last_update > args.frame_timeout:
                logger.debug("Frame timeout for cid=%s (%.3fs)", cid, now - state.last_update)
                state.points = []

            state.last_update = now
            state.last_frame_id = frame_id
            state.points.extend(points)

            if len(state.points) < args.min_points:
                continue

            if len(state.points) > args.max_points:
                state.points = state.points[-args.max_points :]

            total_pts = len(state.points)
            pts = np.asarray(state.points, dtype=np.float32)
            pts = maybe_sample(pts, args.sample_points)

            z_min, z_max, height = robust_height(pts)
            height_corr = corrected_height(height, pts, args.height_corr_coeff)
            dist_xy = float(np.linalg.norm(np.mean(pts[:, :2], axis=0)))

            pts_norm = normalize_points(pts)
            model_input = build_model_input(pts_norm, infer_tf, device)

            with torch.no_grad():
                output = model(model_input)
                logits = output.get("logits", list(output.values())[0]) if isinstance(output, dict) else output
                probs = torch.softmax(logits, dim=-1)[0]
                pred = int(torch.argmax(probs).item())

            logger.info(
                "📍 [ID %s] dist=%.2fm | Hauteur=%.2fm | Corr=%.2fm | Points=%s | Classe=%s (%.1f%%)",
                cid,
                dist_xy,
                height,
                height_corr,
                total_pts,
                args.class_names[pred],
                probs[pred].item() * 100.0,
            )

            if args.debug:
                log_histogram(logger, "coord_x", pts[:, 0])
                log_histogram(logger, "coord_y", pts[:, 1])
                log_histogram(logger, "coord_z", pts[:, 2])
                log_histogram(logger, "logits", logits.detach().cpu().numpy().ravel())

            state.points = []

    except KeyboardInterrupt:
        logger.info("Arrêt.")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
