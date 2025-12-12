import numpy as np
import torch
import pandas as pd
from ..torchrtm.models import prosail

def test_prosail_shell_v2_runs():
    batch_size = 2
    # Dummy leaf traits: [Cab, Car, Cbr, Cw, Cm] per sample.
    traits = torch.tensor([[40.0, 8.0, 0, 0.01, 0.008]] * batch_size)
    N = torch.tensor([1.5] * batch_size)
    # Dummy leaf angle parameters (for LIDF functions)
    LIDFa = torch.tensor([[-0.3]] * batch_size)
    LIDFb = torch.tensor([[-0.1]] * batch_size)
    # Dummy canopy parameters:
    lai = torch.tensor([3.0] * batch_size)    # Leaf Area Index
    q = torch.tensor([0.5] * batch_size)        # Hotspot parameter
    # Geometry (angles in degrees)
    tts = torch.tensor([30.0] * batch_size)     # Solar zenith angle
    tto = torch.tensor([20.0] * batch_size)     # Sensor view zenith angle
    psi = torch.tensor([10.0] * batch_size)       # Relative azimuth angle
    tran_alpha = torch.tensor([40.0] * batch_size)  # Leaf transmission parameter
    # Soil moisture (psoil): a value between 0 and 1 per sample
    psoil = torch.tensor([0.5] * batch_size)

    # Dummy soil spectra for testing (dummy reflectance with 2101 wavelength points):
    # In practice, the model would load actual data. Here we supply plausible dummy values.
    dry_soil = torch.linspace(0.2, 0.3, 2101)
    wet_soil = torch.linspace(0.03, 0.04, 2101)

    # Geometric constants for the model
    rd = torch.tensor(np.pi / 180)
    pi_val = torch.tensor(np.pi)
    # Define leaf angle bins (typically 13 values, from 5 to 89 deg)
    litab = torch.linspace(5, 89, 13)
    
    # Call the high-level PROSAIL shell function.
    result = prosail(
        traits, N, LIDFa, LIDFb, lai, q,
        tts, tto, psi, tran_alpha, psoil,
        batch_size=batch_size, use_prospectd=False, lidtype=1
    )
    rsot = result[3]
    
    # In batch mode, the function should return a torch.Tensor.
    # For example, if the output represents canopy reflectance across 2101 wavelengths,
    # result should have shape (batch_size, 2101) or similar.
    assert isinstance(result, torch.Tensor), "Expected output to be a torch.Tensor in batch mode."
    assert isinstance(rsot, torch.Tensor), "Expected RSOT to be a torch.Tensor"
    assert rsot.shape[0] == 2101, f"Expected 2101 wavelengths along dim 0, got {rsot.shape[0]}"
        # Ensure all output values are finite (no NaNs or Infs)
    assert torch.all(torch.isfinite(result)), "Output contains non-finite values."


