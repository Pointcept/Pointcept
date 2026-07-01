"""Bits2Bites — PointTransformerV3 inference/standalone-eval config (no train hooks).

Run with tools/test.py -> MultiClsTester. Point data_root at the labelled
held-out fold for metrics, or at unlabelled scans for prediction-only output.
"""

_base_ = ["cls-ptv3-base.py"]

hooks = []
