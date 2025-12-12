"""
torchrtm.canopy
---------------

Canopy-level radiative transfer using 4SAIL and related models.

Includes:
    - 4SAIL main interface
    - Geometric and angular scattering modules
    - Hotspot and soil reflectance integration

Authors:
    - Peng Sun
    - Marco D. Visser
"""

from .sail import foursail_main
from .sail_geometry import sail_sensgeom
from .sail_radiative import RTgeom, ReflTrans, sail_BDRF
from .hotspot import HotSpot
from .suits import SUITS
from .lidf import lidf_1, lidf_2, lidf_3, lidf_4
