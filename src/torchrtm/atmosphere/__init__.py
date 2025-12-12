"""
torchrtm.atmosphere
-------------------

Atmospheric correction utilities for radiative transfer modeling.
Includes SMAC correction, pressure estimation, and TOC–TOA conversion logic.
"""

from .smac import (
    smac,
    calculate_pressure_from_altitude,
    canp_to_ban_5,
    toc_to_toa,
)

__all__ = [
    "smac",
    "calculate_pressure_from_altitude",
    "canp_to_ban_5",
    "toc_to_toa",
]
