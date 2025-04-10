"""
torchrtm
--------

Core RTM simulation models for canopy reflectance using PROSAIL and SMAC.

Submodules:
    - leaf: Leaf-level optical models (PROSPECT family)
    - canopy: Canopy radiative transfer (4SAIL and geometric modules)
    - atmosphere: Atmospheric correction models (SMAC)
    - utils: Mathematical and device handling utilities
    - data_loader: Internal dataset access (coefficients, sensors, soil)
    - models: High-level model interfaces (e.g., PROSAIL shell)

Authors:
    - Peng Sun
    - Marco D. Visser
"""

from . import leaf
from . import canopy
from . import atmosphere
from . import utils
from . import data_loader     # Added to expose data_loader as part of the public API
from . import models          # High-level models (only PROSAIL for now)

__version__ = "0.1.0"
