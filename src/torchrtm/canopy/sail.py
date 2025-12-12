"""
torchrtm.canopy.sail
--------------------

Top-level canopy RTM function using the 4SAIL model.
"""

import torch
from torchrtm.canopy.sail_geometry import sail_sensgeom
from torchrtm.canopy.sail_radiative import RTgeom, ReflTrans, sail_BDRF
from torchrtm.canopy.lidf import lidf_1, lidf_2, lidf_3, lidf_4
from torchrtm.utils import to_device
from torchrtm.data_loader import load_soil_spectra


def foursail_main(
    rho, tau, psoil, LIDFa, LIDFb, lai, q,
    tts, tto, psi, rd, pi, litab,
    dry_soil=None, wet_soil=None, lidtype=1
):
    """
    Full 4SAIL RTM canopy simulation.

    Args:
        rho, tau (torch.Tensor): Leaf reflectance & transmittance (wavelength x batch).
        psoil (float or tensor): Soil moisture fraction (0–1).
        LIDFa, LIDFb (float or tensor): Leaf angle parameters.
        lai (tensor): Leaf area index.
        q (tensor): Hotspot parameter.
        tts, tto, psi (float/tensor): Viewing geometry.
        rd (float): Radians per degree.
        pi (float): π constant.
        litab (tensor): Leaf angle bins (13 values).
        dry_soil, wet_soil (torch.Tensor): Soil reflectance spectra.
        use_batch (bool): Enable batched processing.
        lidtype (int): LIDF function selector [1–4].

    Returns:
        List of canopy reflectance and transmittance terms.
    """

      # Fallback to default soil data if none is provided
    if dry_soil is None or wet_soil is None:
        dry_np, wet_np = load_soil_spectra()
        dry_soil = torch.tensor(dry_np, dtype=torch.float32, device=rho.device)
        wet_soil = torch.tensor(wet_np, dtype=torch.float32, device=rho.device)

    na = torch.zeros(len(litab)).to(rho.device)
    len_na = len(na)
    len_batch = lai.shape[0] 

    # Viewing geometry
    cts, cto, ctscto, dso = sail_sensgeom(tts, tto, psi, rd)

    # Soil reflectance
    if psoil.ndim >= 1:
        psoil_set = psoil.expand(dry_soil.shape[0], len_batch).T
        rsoil = (psoil_set * dry_soil + (1 - psoil_set) * wet_soil).T
    else:
        rsoil = psoil * dry_soil + (1 - psoil) * wet_soil

    # Leaf angle distribution
    if lidtype == 1:
        lidFun = lidf_1(na, LIDFa, LIDFb)
    elif lidtype == 2:
        lidFun = lidf_2(na, LIDFa)
    elif lidtype == 3:
        lidFun = lidf_3(na, LIDFa, LIDFb)
    elif lidtype == 4:
        lidFun = lidf_4(na, LIDFa)
    else:
        raise ValueError(f"Unsupported lidtype: {lidtype}")

    # SUITS angular scattering terms (replace this w/ moduleized call if needed)
    from torchrtm.canopy.suits import SUITS
    ks, ko, sob, sof, sdb, sdf, dob, dof, ddb, ddf = SUITS(
        na, litab, lidFun, tts, tto, cts, cto, psi, ctscto, len_batch=len_batch, len_na=len_na
    )

    # Leaf reflectance modeling
    sigb, att, m, sb, sf, vb, vf, w = RTgeom(rho.T, tau.T, ddb, ddf, sdb, sdf, dob, dof, sob, sof)

    rdd, tdd, tsd, rsd, tdo, rdo, tss, too, rsod = ReflTrans(
        rho, tau, lai, att, m, sigb, ks, ko, sf, sb, vf, vb
    )

    # Hotspot effect
    from torchrtm.canopy.hotspot import HotSpot
    tsstoo, sumint = HotSpot(lai, q, tss, ks, ko, dso)

    # Final BRDF computation
    finalOut = sail_BDRF(
        w, lai, sumint, tsstoo, rsoil,
        rdd, tdd, tsd, rsd, tdo, rdo, tss, too, rsod,
        len_batch=len_batch
    )

    # Transpose outputs to match expected (B, 2101) shape
    finalOut = [x.T if x.ndim == 2 and x.shape[0] == 2101 else x for x in finalOut]
    return finalOut

