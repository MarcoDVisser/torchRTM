"""
tests/test_atmosphere.py
-------------------------

Unit tests for atmospheric correction utilities and SMAC–PROSAIL linkage.
"""

import torch
import pytest
from ..torchrtm.atmosphere.smac import (
    smac,
    calculate_pressure_from_altitude,
    canp_to_ban_5,
    toc_to_toa,
)
from ..torchrtm.models import prosail_shell_v2
from ..torchrtm.data_loader import load_coefmat, load_prospectd_matrix, load_soil_spectra, load_smac_sensor


def test_pressure_from_altitude():
    altitudes = torch.tensor([0.0, 1000.0, 5000.0])
    pressures = calculate_pressure_from_altitude(altitudes,271.0)
    assert pressures.shape == altitudes.shape, "Output shape mismatch."
    assert torch.all(pressures > 0), "All pressures must be positive."
    assert pressures[0] > pressures[-1], "Pressure should decrease with altitude."


def test_canp_to_ban_5_shape():
    # Simulate TOC-like input: shape [4, B, WL]
    B, WL = 3, 2101
    toc = torch.rand(4, B, WL)
    sm_wl = torch.tensor([100, 500, 1000])
    result = canp_to_ban_5(toc, sm_wl)
    assert len(result) == 4, "Expected 4 outputs from canp_to_ban_5"
    for r in result:
        assert r.shape == (B, len(sm_wl)), f"Unexpected shape {r.shape}"


def test_toc_to_toa_output_shape_only():
    # Simulate full TOC input: shape [4, B, WL]
    B, WL = 2, 2101
    toc = torch.rand(4, B, WL)
    sm_wl = torch.tensor([10, 20])

    ta = torch.ones(B)
    rtoa = toc_to_toa(toc, sm_wl, ta, ta, ta, ta, ta, ta, ta)

    assert rtoa.shape == (B, len(sm_wl)), f"Expected shape {(B, len(sm_wl))}, got {rtoa.shape}"
    assert torch.all(rtoa >= 0), "TOA reflectance should be non-negative."


def test_toc_to_toa_with_prosail():
    """
    Integration test: Ensure PROSAIL output can be combined with SMAC to compute TOA reflectance.
    """
    B = 2
    WL = 2101
    sm_wl = torch.arange(WL)  # Use all wavelengths

    # Simulate canopy reflectance with PROSAIL
    toc_full = prosail_shell_v2(
        traits=torch.tensor([[40.0, 8.0, 0.0, 0.01, 0.008]] * B),
        N=torch.tensor([1.5] * B),
        LIDFa=torch.tensor([[-0.3]] * B),
        LIDFb=torch.tensor([[-0.1]] * B),
        lai=torch.tensor([3.0] * B),
        q=torch.tensor([0.5] * B),
        tts=torch.tensor([30.0] * B),
        tto=torch.tensor([20.0] * B),
        psi=torch.tensor([10.0] * B),
        tran_alpha=torch.tensor([40.0] * B),
        psoil=torch.tensor([0.5] * B),
        batch_size=B,
        use_prospectd=False,
        lidtype=1
    )

    # Convert RTM output [WL, B] → [B, WL] for TOA function
    toc = [x.T for x in toc_full[0:4]]

    # Get atmospheric correction coefficients
    tts = torch.tensor([30.0] * B)
    tto = torch.tensor([20.0] * B)
    psi = torch.tensor([10.0] * B)

    # Load sensor information
    sensor_name = "Sentinel2A-MSI"
    coefs,sm_wl = load_smac_sensor(sensor_name.split('.')[0])
    bans_num =  len(sm_wl)
    # Run the SMAC function with the loaded coefficients
    outputs = smac(tts, tto, psi, coefs)
    # Add assertions to check that the outputs are as expected
    assert outputs[0].shape == (B, bans_num), "ttetas has incorrect shape"
    assert outputs[1].shape == (B, bans_num), "ttetav has incorrect shape"
    assert outputs[2].shape == (B, bans_num), "tg has incorrect shape"
    assert outputs[3].shape == (B, bans_num), "s has incorrect shape"
    assert outputs[4].shape == (B, bans_num), "atm_ref has incorrect shape"
    assert outputs[5].shape == (B, bans_num), "tdir_tts has incorrect shape"
    assert outputs[6].shape == (B, bans_num), "tdif_tts has incorrect shape"
    assert outputs[7].shape == (B, bans_num), "tdir_ttv has incorrect shape"
    assert outputs[8].shape == (B, bans_num), "tdif_ttv has incorrect shape"

    assert torch.all(outputs[5] >= 0), "All TOA reflectance values (tdir_tts) should be non-negative."
    assert torch.all(outputs[6] >= 0), "All TOA reflectance values (tdif_tts) should be non-negative."
    assert torch.all(outputs[7] >= 0), "All TOA reflectance values (tdir_ttv) should be non-negative."
    assert torch.all(outputs[8] >= 0), "All TOA reflectance values (tdif_ttv) should be non-negative."