"""Bits2Bites — ptv3 single-task learning (STL): head 0 (right sagittal) only."""

_base_ = ["cls-ptv3-base.py"]

model = dict(stl_task=0)
