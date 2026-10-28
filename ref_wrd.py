"""Canonical ReF-WRD entry points.

The implementation remains in fde_sr.py so existing experiment scripts and
checkpoints keep working. New code can import the paper-aligned names here.
"""

from fde_sr import (
    ReF_WRD_gen,
    ReF_WRD_inference_time,
    ReF_WRD_reg,
    ReF_WRD_test,
)


__all__ = [
    "ReF_WRD_gen",
    "ReF_WRD_inference_time",
    "ReF_WRD_reg",
    "ReF_WRD_test",
]
