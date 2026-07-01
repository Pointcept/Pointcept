"""Bits2Bites — spunet single-task learning (STL): head 1 (left sagittal) only."""

_base_ = ["cls-spunet-base.py"]

model = dict(stl_task=1)
