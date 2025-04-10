"""
torchrtm.canopy.suits
---------------------

Implements the SUITS directional scattering approximation used in SAIL.
Includes the volume scattering functions used in SUITS approximation for canopy radiative transfer.
"""

import torch


def SUITS(na, litab, lidFun, tts, tto, cts, cto, psi, ctscto,
          use_batch=True, ks=0, ko=0, bf=0, sob=0, sof=0,
          len_batch=0, len_na=13):
    """
    Computes scattering parameters for SAIL using the SUITS approximation.

    Args:
        na (torch.Tensor): Dummy tensor used for dimension tracking (e.g., angular bins).
        litab (torch.Tensor): Leaf inclination bin centers [deg].
        lidFun (torch.Tensor): LIDF weights, shape (B, 13) or (13,).
        tts (torch.Tensor): Solar zenith angle [deg], shape (B,) or scalar.
        tto (torch.Tensor): Observer zenith angle [deg], shape (B,) or scalar.
        cts (torch.Tensor): cos(tts), shape (B,) or scalar.
        cto (torch.Tensor): cos(tto), shape (B,) or scalar.
        psi (torch.Tensor): Relative azimuth angle [deg], shape (B,) or scalar.
        ctscto (torch.Tensor): cos(tts) * cos(tto), shape (B,) or scalar.
        use_batch (bool): Whether to apply batched processing.
        ks, ko, bf, sob, sof (float): Initial scattering terms.
        len_batch (int): Batch size (only required if use_batch=True).
        len_na (int): Number of angular bins (default 13).

    Returns:
        list: [ks, ko, sob, sof, sdb, sdf, dob, dof, ddb, ddf]
    """
    pi = torch.tensor(torch.pi, device=cts.device)
    rd = pi / 180.0
    ctl = torch.cos(rd * litab)

    chi_s, chi_o, frho, ftau = volscatt(tts, tto, psi, litab, len_batch, len_na)

    if use_batch:
        ksli = chi_s / cts.unsqueeze(1)
        koli = chi_o / cto.unsqueeze(1)
        sobli = frho * pi / ctscto.unsqueeze(1)
        sofli = ftau * pi / ctscto.unsqueeze(1)
        bfli = ctl**2

        ks = (ksli * lidFun).sum(dim=1)
        ko = (koli * lidFun).sum(dim=1)
        bf = (bfli.unsqueeze(0) * lidFun).sum(dim=1)
        sob = (sobli * lidFun).sum(dim=1)
        sof = (sofli * lidFun).sum(dim=1)
    else:
        ks = torch.dot(chi_s / cts, lidFun)
        ko = torch.dot(chi_o / cto, lidFun)
        bf = torch.dot(ctl**2, lidFun)
        sob = torch.dot(frho * pi / ctscto, lidFun)
        sof = torch.dot(ftau * pi / ctscto, lidFun)

    sdb = 0.5 * (ks + bf)
    sdf = 0.5 * (ks - bf)
    dob = 0.5 * (ko + bf)
    dof = 0.5 * (ko - bf)
    ddb = 0.5 * (1 + bf)
    ddf = 0.5 * (1 - bf)

    return [ks, ko, sob, sof, sdb, sdf, dob, dof, ddb, ddf]


def volscatt(tts, tto, psi, ttl, len_batch=0, len_na=13):
    """
    Computes volume scattering functions (Verhoef 1984) for SUITS model.

    Args:
        tts (torch.Tensor): Solar zenith angle [deg], shape (B,) or scalar.
        tto (torch.Tensor): Observer zenith angle [deg], shape (B,) or scalar.
        psi (torch.Tensor): Relative azimuth angle [deg], shape (B,) or scalar.
        ttl (torch.Tensor): Leaf inclination bin centers [deg], shape (13,).
        len_batch (int): Batch size (used only when batched).
        len_na (int): Number of angle bins (default 13).

    Returns:
        tuple: chi_s, chi_o, frho, ftau – all shape (B, 13) or (13,)
    """
    pi = torch.tensor(torch.pi, device=ttl.device)
    rd = pi / 180.0

    ctl = torch.cos(rd * ttl)              # (13,)
    stl = torch.sin(rd * ttl)

    costs = torch.cos(rd * tts)            # (B,)
    sints = torch.sin(rd * tts)
    costo = torch.cos(rd * tto)
    sinto = torch.sin(rd * tto)
    cospsi = torch.cos(rd * psi)

    # Ensure correct shape for broadcasting: (B, 13)
    cs = costs.unsqueeze(1) * ctl.unsqueeze(0)
    ss = sints.unsqueeze(1) * stl.unsqueeze(0)
    co = costo.unsqueeze(1) * ctl.unsqueeze(0)
    so = sinto.unsqueeze(1) * stl.unsqueeze(0)

    # Clamp to avoid numerical instability
    cosbts = torch.clamp(-cs / (ss + 1e-6), -1, 1)
    cosbto = torch.clamp(-co / (so + 1e-6), -1, 1)

    bts = torch.acos(cosbts)
    bto = torch.acos(cosbto)

    chi_s = 2 / pi * ((bts - pi / 2) * cs + torch.sin(bts) * ss)
    chi_o = 2 / pi * ((bto - pi / 2) * co + torch.sin(bto) * so)

    frho = torch.abs(bts - bto)
    ftau = pi - torch.abs(bts + bto - pi)

    return chi_s, chi_o, frho, ftau
