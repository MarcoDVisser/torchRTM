"""
torchrtm.atmosphere.smac
------------------------

Full SMAC model with seamless integration into torchrtm, supporting GPU inference
and automatic loading of sensor-specific atmospheric coefficients.
"""

import torch
from torchrtm.utils import to_device
from torchrtm.data_loader import load_smac_sensor
import numpy as np
import pickle
import importlib.resources as pkg_resources
import torchrtm.data.sensor_information as sensor_pkg
import torch
from torchrtm.utils import to_device


def smac(tts, tto, psi, coefs, aot550=0.325, uo3=0.35, uh2o=1.41, Pa=1013.25):
    """
    Calculates atmospheric radiative transfer using the PROSAIL model.
    This function computes various components such as atmospheric transmission, reflectance, and scattering
    for given angles and aerosol properties.

    Parameters:
        tts (tensor): Solar zenith angle (in degrees) - angle between the sun and the vertical.
        tto (tensor): View zenith angle (in degrees) - angle between the sensor and the vertical.
        psi (tensor): Relative azimuth angle (in degrees) between the solar and viewing directions.
        coefs (dict): Coefficients for the atmospheric and aerosol properties.
        aot550 (float): Aerosol optical thickness at 550 nm (default = 0.325).
        uo3 (float): Ozone absorption coefficient.
        uh2o (float): Water vapor absorption coefficient.
        Pa (float): Atmospheric pressure (default = 1013.25 hPa).

    Returns:
        tuple: Atmospheric radiative transfer components such as reflectance, scattering, and transmission.
    """

    def to_cuda(x):
      """Converts data to a CUDA tensor."""
      return torch.tensor(x, dtype=torch.float32).to(device)
    device = tts.device
    pi = torch.tensor(3.14159265).to(device)
    taup550 = to_cuda(aot550)  # Aerosol optical thickness at 550 nm.

    # Convert constants to tensors if they are not already.
    uo3 = to_cuda(uo3)
    uh2o = to_cuda(uh2o)
    bans_num = coefs["a0taup"].shape[-1]
    if len(tts.shape) >0:
      batchsize = tts.shape[0]
      taup550 = taup550.expand(bans_num,batchsize).T
      uo3 = uo3.expand(bans_num,batchsize).T
      uh2o = uh2o.expand(bans_num,batchsize).T
      tts = tts.expand(bans_num,batchsize).T
      tto = tto.expand(bans_num,batchsize).T
      psi = psi.expand(bans_num,batchsize).T

    # Extract coefficients from the provided dictionary.
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

    # Conversion constants from degrees to radians and vice versa.
    cdr = pi / 180  # Degrees to radians conversion factor.
    crd = 180 / pi  # Radians to degrees conversion factor.

    # Calculate solar and view zenith angles.
    us = torch.cos(tts * cdr)  # Solar zenith angle cosine.
    uv = torch.cos(tto * cdr)  # View zenith angle cosine.
    Peq = Pa / 1013.25  # Pressure ratio.

    # Air mass computation (used in various aerosol and gaseous transmission calculations).
    m = 1 / us + 1 / uv
    # Aerosol optical depth computation for the given wavelengths.
    taup = a0taup + a1taup * taup550

    # Transmission values for various gases based on pressure.
    uo2 = Peq ** po2
    uco2 = Peq ** pco2
    uch4 = Peq ** pch4
    uno2 = Peq ** pno2
    uco = Peq ** pco

    # Compute the transmission for water vapor, ozone, and other gases.
    to3 = torch.exp(ao3 * (uo3 * m) ** no3)
    th2o = torch.exp(ah2o * (uh2o * m) ** nh2o)
    to2 = torch.exp(ao2 * (uo2 * m) ** no2)
    tco2 = torch.exp(aco2 * (uco2 * m) ** nco2)
    tch4 = torch.exp(ach4 * (uch4 * m) ** nch4)
    tno2 = torch.exp(ano2 * (uno2 * m) ** nno2)
    tco = torch.exp(aco * (uco * m) ** nco)

    # Total atmospheric transmission as a product of various gaseous transmissions.
    tg = th2o * to3 * to2 * tco2 * tch4 * tco * tno2

    # Calculate the spherical albedo of the atmosphere (accounting for aerosol properties).
    s = a0s * Peq + a3s + a1s * taup550 + a2s * taup550 ** 2

    # Scattering transmission for the total atmosphere in both zenith and view directions.
    ttetas = a0T + a1T * taup550 / us + (a2T * Peq + a3T) / (1 + us)
    ttetav = a0T + a1T * taup550 / uv + (a2T * Peq + a3T) / (1 + uv)

    # Calculate the cosine of the scattering angle between the solar and view directions.
    cksi = -(
        (us * uv) + (torch.sqrt(1 - us * us) * torch.sqrt(1 - uv * uv) * torch.cos(psi * crd))
    )

    # Ensure the cosine of the scattering angle is within valid bounds.
    cksi[cksi < -1] = -1  # Ensures no values are below -1, as the arccos function cannot handle this.

    # Scattering angle in degrees.
    ksiD = crd * torch.arccos(cksi)

    # Rayleigh scattering phase function and reflectance.
    ray_phase = 0.7190443 * (1 + (cksi * cksi)) + 0.0412742
    ray_ref = (taur * ray_phase) / (4 * us * uv)
    ray_ref = ray_ref * Pa / 1013.25  # Correct for pressure variation.

    taurz = taur * Peq  # Eq 12

    # Aerosol phase function using polynomial expansion.
    aer_phase = (
        a0P + a1P * ksiD + a2P * ksiD * ksiD + a3P * ksiD ** 3 + a4P * ksiD ** 4
    )
    ak2 = (1 - wo) * (3 - wo * 3 * gc)
    ak = torch.sqrt(ak2)

    e = -3 * us * us * wo / (4 * (1 - ak2 * us * us))
    f = -(1 - wo) * 3 * gc * us * us * wo / (4 * (1 - ak2 * us * us))
    dp = e / (3 * us) + us * f
    d = e + f
    b = 2 * ak / (3 - wo * 3 * gc)
    delta = torch.exp(ak * taup) * (1 + b) ** 2 - torch.exp(-ak * taup) * (1 - b) ** 2
    ww = wo / 4
    ss = us / (1 - ak2 * us * us)
    q1 = 2 + 3 * us + (1 - wo) * 3 * gc * us * (1 + 2 * us)
    q2 = 2 - 3 * us - (1 - wo) * 3 * gc * us * (1 - 2 * us)
    q3 = q2 * torch.exp(-taup / us)
    c1 = ((ww * ss) / delta) * (q1 * torch.exp(ak * taup) * (1 + b) + q3 * (1 - b))
    c2 = -((ww * ss) / delta) * (q1 * torch.exp(-ak * taup) * (1 - b) + q3 * (1 + b))
    cp1 = c1 * ak / (3 - wo * 3 * gc)
    cp2 = -c2 * ak / (3 - wo * 3 * gc)
    z = d - wo * 3 * gc * uv * dp + wo * aer_phase / 4
    x = c1 - wo * 3 * gc * uv * cp1
    y = c2 - wo * 3 * gc * uv * cp2
    aa1 = uv / (1 + ak * uv)
    aa2 = uv / (1 - ak * uv)
    aa3 = us * uv / (us + uv)

    # Final computation of the aerosol and residual atmospheric reflectance.
    aer_ref1 = x * aa1 * (1 - torch.exp(-taup / aa1))  # Equation for aerosol reflectance.
    aer_ref2 = y * aa2 * (1 - torch.exp(-taup / aa2))
    aer_ref3 = z * aa3 * (1 - torch.exp(-taup / aa3))

    aer_ref = (aer_ref1 + aer_ref2 + aer_ref3) / (us * uv)  # Total aerosol reflectance.

    # Additional calculations for the residual atmospheric contributions.
    Res_ray = Resr1 + Resr2 * taur * ray_phase / (us * uv) + Resr3 * ((taur * ray_phase / (us * uv)) ** 2)

    tautot = taup + taurz

    Res_aer = (Resa1 + Resa2 * (taup * m * cksi) + Resa3 * ((taup * m * cksi) ** 2)) + Resa4 * (taup * m * cksi) ** 3
    Res_6s = (Rest1 + Rest2 * (tautot * m * cksi) + Rest3 * ((tautot * m * cksi) ** 2)) + Rest4 * ((tautot * m * cksi) ** 3)

    # Total atmospheric reflectance considering all components.
    atm_ref = ray_ref - Res_ray + aer_ref - Res_aer + Res_6s

    # For non-Lambertian surface, compute direct and diffuse components.
    tdir_tts = torch.exp(-tautot / us)
    tdir_ttv = torch.exp(-tautot / uv)
    tdif_tts = ttetas - tdir_tts
    tdif_ttv = ttetav - tdir_ttv

    # Return calculated components.
    return (ttetas, ttetav, tg, s, atm_ref, tdir_tts, tdif_tts, tdir_ttv, tdif_ttv)
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
'''def toc_to_toa(toc, sm_wl, ta_ss, ta_sd, ta_oo, ta_do, ra_so, ra_dd, T_g, return_toc=False):
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

    return (R_TOC, R_TOA) if return_toc else R_TOA'''

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
    #rddt, rsdt, rdot, rsot = canp_to_ban_5(toc, sm_wl)
    rv_dd,rv_sd,rv_so,rv_do = canp_to_ban_5(toc, sm_wl)
    rtoa0 = ra_so + ta_ss * rv_so * ta_oo  # Reflectance component for soil (ra_so) and other terms
    rtoa1 = (
        (ta_sd * rv_do + ta_ss * rv_sd * ra_dd * rv_do)  # Reflectance component for vegetation to sensor
        * ta_oo
        / (1 - rv_dd * ra_dd)  # Adjusting for the soil-vegetation interaction
    )
    rtoa2 = (ta_ss * rv_sd + ta_sd * rv_dd) * ta_do / (1 - rv_dd * ra_dd)  # Another component for vegetation

    # Top-of-Canopy (TOC) reflectance calculation
    R_TOC = (ta_ss * rv_so + ta_sd * rv_do) / (ta_ss + ta_sd)  # Weighted sum of soil and vegetation components

    # Final TOA reflectance calculation using the total radiance terms
    R_TOA = T_g * (rtoa0 + rtoa1 + rtoa2)  # Final TOA reflectance, including the gain factor (T_g)


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


def get_sensor_file_list(keyword: str):
    # 获取所有资源名（即文件名）
    files = [f.name for f in pkg_resources.files(sensor_pkg).iterdir() if f.name.endswith('.pkl')]

    # 根据关键词过滤
    matched = sorted([f for f in files if keyword in f])

    # 构建全路径（可以用 open 读取）
    full_paths = [pkg_resources.files(sensor_pkg).joinpath(f) for f in matched]
    return np.array(full_paths)