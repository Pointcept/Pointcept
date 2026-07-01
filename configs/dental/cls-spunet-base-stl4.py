"""Bits2Bites — spunet single-task learning (STL): head 4 (midline) only."""

_base_ = ["cls-spunet-base.py"]

model = dict(stl_task=4)
