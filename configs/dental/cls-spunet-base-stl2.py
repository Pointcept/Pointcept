"""Bits2Bites — spunet single-task learning (STL): head 2 (anterior bite) only."""

_base_ = ["cls-spunet-base.py"]

model = dict(stl_task=2)
