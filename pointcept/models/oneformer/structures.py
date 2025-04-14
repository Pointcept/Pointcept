"""
# This file includes code adapted from OneFormer:
# https://github.com/SHI-Labs/OneFormer
"""

from .instance_data_og import InstanceData
from collections.abc import Sized


class InstanceData_(InstanceData):
    """We only remove a single assert from __setattr__."""

    def __setattr__(self, name: str, value: Sized):
        """setattr is only used to set data.

        The value must have the attribute of `__len__` and have the same length
        of `InstanceData`.
        """
        if name in ("_metainfo_fields", "_data_fields"):
            if not hasattr(self, name):
                super(InstanceData, self).__setattr__(name, value)
            else:
                raise AttributeError(
                    f"{name} has been used as a "
                    "private attribute, which is immutable."
                )

        else:
            assert isinstance(value, Sized), "value must contain `__len__` attribute"

            super(InstanceData, self).__setattr__(name, value)
