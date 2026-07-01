"""Bits2Bites — spunet single-task learning (STL): head 0 (right sagittal) only."""

_base_ = ["cls-spunet-base.py"]

model = dict(stl_task=0)
