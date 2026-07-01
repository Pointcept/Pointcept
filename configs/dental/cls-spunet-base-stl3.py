"""Bits2Bites — spunet single-task learning (STL): head 3 (transverse) only."""

_base_ = ["cls-spunet-base.py"]

model = dict(stl_task=3)
