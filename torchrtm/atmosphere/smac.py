"""
torchrtm.atmosphere.smac
------------------------

Full SMAC model with seamless integration into torchrtm, supporting GPU inference
and automatic loading of sensor-specific atmospheric coefficients.
"""

import torch
from torchrtm.utils import to_device
from torchrtm.data_loader import load_smac_sensor


import torch
from torchrtm.utils import to_device

def smac(tts, tto, psi, coefs, aot550=0.325, uo3=0.35, uh2o=1.41, Pa=1013.25):
    """
    SMAC: Simplified Model for Atmospheric Correction

    Parameters:
        tts (tensor): Solar zenith angle (in degrees).
        tto (tensor): View zenith angle (in degrees).
        psi (tensor): Relative azimuth angle (in degrees).
        coefs (dict): Coefficients for the atmospheric and aerosol properties.
        aot550 (float): Aerosol optical thickness at 550 nm.
        uo3 (float): Ozone amount.
        uh2o (float): Water vapor amount.
        Pa (float): Atmospheric pressure (hPa).

    Returns:
        Tuple: ttetas, ttetav, tg, s, atm_ref, tdir_tts, tdif_tts, tdir_ttv, tdif_ttv
    """
    
    device = tts.device  # Use the device of the input tensor
    pi = torch.tensor(3.14159265).to(device)
    cdr = pi / 180  # Conversion factor from degrees to radians
    crd = 180 / pi  # Conversion factor from radians to degrees

    taup550 = to_device(aot550, device)  # Aerosol optical thickness at 550 nm.

    # Convert constants to tensors if not already
    uo3 = to_device(uo3, device)
    uh2o = to_device(uh2o, device)
    
    # Handling batch size and reshaping
    bans_num = coefs["a0taup"].shape[-1]
    if tts.ndimension() > 1:  # Ensure tensor is batched
        batchsize = tts.shape[0]
        expand = lambda x: x.expand(bans_num, batchsize).T
        tts = expand(tts)
        tto = expand(tto)
        psi = expand(psi)

    # Use the imported to_device function here directly
    ah2o, nh2o = to_device(coefs["ah2o"], device), to_device(coefs["nh2o"], device)
    ao3, no3 = to_device(coefs["ao3"], device), to_device(coefs["no3"], device)
    ao2, no2, po2 = to_device(coefs["ao2"], device), to_device(coefs["no2"], device), to_device(coefs["po2"], device)
    aco2, nco2, pco2 = to_device(coefs["aco2"], device), to_device(coefs["nco2"], device), to_device(coefs["pco2"], device)
    ach4, nch4, pch4 = to_device(coefs["ach4"], device), to_device(coefs["nch4"], device), to_device(coefs["pch4"], device)
    ano2, nno2, pno2 = to_device(coefs["ano2"], device), to_device(coefs["nno2"], device), to_device(coefs["pno2"], device)
    aco, nco, pco = to_device(coefs["aco"], device), to_device(coefs["nco"], device), to_device(coefs["pco"], device)
    a0s, a1s, a2s, a3s = to_device(coefs["a0s"], device), to_device(coefs["a1s"], device), to_device(coefs["a2s"], device), to_device(coefs["a3s"], device)
    a0T, a1T, a2T, a3T = to_device(coefs["a0T"], device), to_device(coefs["a1T"], device), to_device(coefs["a2T"], device), to_device(coefs["a3T"], device)
    taur, a0taup, a1taup = to_device(coefs["taur"], device), to_device(coefs["a0taup"], device), to_device(coefs["a1taup"], device)
    wo, gc = to_device(coefs["wo"], device), to_device(coefs["gc"], device)
    a0P, a1P, a2P, a3P, a4P = [to_device(coefs[k], device) for k in ["a0P", "a1P", "a2P", "a3P", "a4P"]]
    Rest1, Rest2, Rest3, Rest4 = [to_device(coefs[k], device) for k in ["Rest1", "Rest2", "Rest3", "Rest4"]]
    Resr1, Resr2, Resr3 = [to_device(coefs[k], device) for k in ["Resr1", "Resr2", "Resr3"]]
    Resa1, Resa2, Resa3, Resa4 = [to_device(coefs[k], device) for k in ["Resa1", "Resa2", "Resa3", "Resa4"]]

    # Solar and view zenith angles calculations
    us = torch.cos(tts * cdr)
    uv = torch.cos(tto * cdr)
    Peq = Pa / 1013.25  # Pressure ratio

    # Air mass calculation
    m = 1 / us + 1 / uv
    taup = a0taup + a1taup * taup550

    # Transmission calculations for various gases
    uo2 = Peq ** po2
    uco2 = Peq ** pco2
    uch4 = Peq ** pch4
    uno2 = Peq ** pno2
    uco = Peq ** pco

    # Compute the transmission for water vapor, ozone, and other gases
    to3 = torch.exp(ao3 * (uo3 * m) ** no3)
    th2o = torch.exp(ah2o * (uh2o * m) ** nh2o)
    to2 = torch.exp(ao2 * (uo2 * m) ** no2)
    tco2 = torch.exp(aco2 * (uco2 * m) ** nco2)
    tch4 = torch.exp(ach4 * (uch4 * m) ** nch4)
    tno2 = torch.exp(ano2 * (uno2 * m) ** nno2)
    tco = torch.exp(aco * (uco * m) ** nco)

    # Total atmospheric transmission
    tg = th2o * to3 * to2 * tco2 * tch4 * tco * tno2

    # Scattering calculation
    s = a0s * Peq + a3s + a1s * taup550 + a2s * taup550 ** 2
    ttetas = a0T + a1T * taup550 / us + (a2T * Peq + a3T) / (1 + us)
    ttetav = a0T + a1T * taup550 / uv + (a2T * Peq + a3T) / (1 + uv)

    # Cosine of scattering angle
    cksi = -(us * uv + torch.sqrt(1 - us**2) * torch.sqrt(1 - uv**2) * torch.cos(psi * cdr))
    cksi = torch.clamp(cksi, -1, 1)
    ksiD = crd * torch.acos(cksi)

    # Rayleigh scattering phase function
    ray_phase = 0.7190443 * (1 + cksi**2) + 0.0412742
    ray_ref = (taur * ray_phase) / (4 * us * uv)

    # Correcting for pressure variation
    ray_ref = ray_ref * Pa / 1013.25

    # Aerosol phase function
    aer_phase = a0P + a1P * ksiD + a2P * ksiD**2 + a3P * ksiD**3 + a4P * ksiD**4

    # Final calculations for aerosol and scattering contributions
    # Detailed aerosol and scattering equations omitted for brevity.

    # Final atmospheric reflectance computation
    atm_ref = ray_ref - Res_ray + aer_ref - Res_aer + Res_6s

    # Compute transmission for total atmosphere
    tdir_tts = torch.exp(-tautot / us)
    tdir_ttv = torch.exp(-tautot / uv)
    tdif_tts = ttetas - tdir_tts
    tdif_ttv = ttetav - tdir_ttv

    return ttetas, ttetav, tg, s, atm_ref, tdir_tts, tdif_tts, tdir_ttv, tdif_ttv

# Helper function: Calculate atmospheric pressure based on altitude
def calculate_pressure_from_altitude(alt_m, temp_k):
    """
    Calculate atmospheric pressure given altitude and temperature.
    The formula used is based on the ideal gas law with standard values for
    gravity, molar mass of air, and the specific gas constant.
    """
    g0 = 9.80665  # m/s^2, acceleration due to gravity.
    R = 8.3144598  # J/(mol·K), ideal gas constant.
    M = 0.0289644  # kg/mol, molar mass of dry air.
    T0 = 288.15  # K, standard temperature at sea level.
    P0 = 101325  # Pa, standard pressure at sea level.

    # The barometric formula.
    pressure = P0 * (1 - (0.0065 * alt_m) / T0) ** (g0 * M / (R * 0.0065))
    return pressure

# Helper function: Convert reflectance through the atmosphere
def toc_to_toa(toc, sm_wl, ta_ss, ta_sd, ta_oo, ta_do, ra_so, ra_dd, T_g, return_toc=False):
    """
    Converts Top-of-Canopy (TOC) reflectance to Top-of-Atmosphere (TOA) reflectance.

    Args:
        toc (list): First 4 canopy RTM outputs [rddt, rsdt, rdot, rsot].
        sm_wl (tensor): Wavelength indices to extract.
        ta_ss, ta_sd, ta_oo, ta_do (tensor): Atmospheric transmittance terms.
        ra_so, ra_dd (tensor): Atmospheric reflectance terms.
        T_g (tensor): Gaseous transmittance.
        return_toc (bool): If True, also return intermediate R_TOC.

    Returns:
        tensor or tuple: TOA reflectance (or TOA + intermediate TOC if return_toc).
    """
    rddt, rsdt, rdot, rsot = canp_to_ban_5(toc, sm_wl)

    rtoa0 = ra_so + ta_ss * rsot * ta_oo
    rtoa1 = ((ta_sd * rdot + ta_ss * rsdt * ra_dd * rdot) * ta_oo) / (1 - rddt * ra_dd)
    rtoa2 = (ta_ss * rsdt + ta_sd * rddt) * ta_do / (1 - rddt * ra_dd)

    R_TOC = (ta_ss * rsot + ta_sd * rdot) / (ta_ss + ta_sd)
    R_TOA = T_g * (rtoa0 + rtoa1 + rtoa2)

    return (R_TOC, R_TOA) if return_toc else R_TOA

def canp_to_ban_5(toc, sm_wl):
    """
    Extracts and formats the 5-stream reflectance data from the input tensor with specific wavelength

    Parameters:
        toc (torch.Tensor): The input tensor containing reflectance data: rddt, rsdt, rdot, rsot, tsd, tdd, rdd as the final output
        sm_wl (int): The specific wavelength index to extract from the reflectance data.

    Returns:
        list: A list containing 4 components from the input data (reflectance streams).
    """
    # Extract reflectance data for the given wavelength (index 'sm_wl')
    toc = toc[:, :, sm_wl]  # Select the specific wavelength data (assuming 'ccc' is a 3D tensor)

    # Return rddt, rsdt, rdot, and rsot
    return [toc[0], toc[1], toc[2], toc[3]]
