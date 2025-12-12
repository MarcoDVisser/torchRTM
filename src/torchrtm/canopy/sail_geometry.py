"""
torchrtm.canopy.sail_geometry
-----------------------------

Computes solar–viewing geometry for the SAIL model.
"""

import torch


def sail_sensgeom(tts, tto, psi, rd):
    """
    Computes geometric factors based on sun and observer angles.

    Args:
        tts (float or torch.Tensor): Solar zenith angle [deg].
        tto (float or torch.Tensor): Observer zenith angle [deg].
        psi (float or torch.Tensor): Relative azimuth angle [deg].
        rd (float or torch.Tensor): Degree-to-radian conversion factor (π/180).

    Returns:
        list: [cos(theta_s), cos(theta_o), cos(theta_s)*cos(theta_o), dso]
    """
    cts = torch.cos(rd * tts)
    cto = torch.cos(rd * tto)
    ctscto = cts * cto

    tants = torch.tan(rd * tts)
    tanto = torch.tan(rd * tto)
    cospsi = torch.cos(rd * psi)

    dso = torch.sqrt(tants ** 2 + tanto ** 2 - 2 * tants * tanto * cospsi)
    return [cts, cto, ctscto, dso]
