"""Bits2Bites — ptv3 single-task learning (STL): head 3 (transverse) only."""

_base_ = ["cls-ptv3-base.py"]

model = dict(stl_task=3)
