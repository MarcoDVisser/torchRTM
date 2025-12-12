import torch
from ..torchrtm.leaf.prospect import prospect5b, prospectd
from ..torchrtm.data_loader import load_coefmat, load_prospectd_matrix, load_soil_spectra, load_smac_sensor

def test_prospect5b_output_shape():
    # Create dummy inputs for a single sample.
    # For PROSPECT-D (test_models.py and test_prospect.py)
    traits = torch.tensor([[40.0, 8.0, 1.0, 0.01, 0.008]])  # [Cab, Car, Cbrown, Cw, Cm]
    N = torch.tensor([1.5])
    # Let the function load the coefficient matrix from data by passing CoefMat=None.
    rho, tau = prospect5b(traits, N, CoefMat=None, device="cpu")
    # Check that output reflectance and transmittance have the expected wavelength dimension (2101)
    assert rho.shape[1] == 2101, f"Expected 2101 wavelengths, got {rho.shape[1]}"
    assert tau.shape[1] == 2101, f"Expected 2101 wavelengths, got {tau.shape[1]}"
    # Check that values are within physically plausible bounds, e.g., between 0 and 1.
    assert torch.all((rho >= 0) & (rho <= 1)), "Reflectance out of bounds"
    assert torch.all((tau >= 0) & (tau <= 1)), "Transmittance out of bounds"


    ## Need prospectd COEF data (not in package)!
#def test_prospectd_bounds():
#    # For PROSPECT-D, use CoefMat=None so that the function loads the default from data.
#    traits = torch.tensor([[40.0, 8.0, 1.0, 0.01, 0.008, 0.0, 0.05]]) # [Cab, Car, Canth, Cbrown, Cw, Cm]
#    N = torch.tensor([1.7])
#    rho, tau = prospectd(traits, N, CoefMat=None, device="cpu")
#    assert rho.shape[1] == 2101, f"Expected 2101 wavelengths, got {rho.shape[1]}"
#    assert tau.shape[1] == 2101, f"Expected 2101 wavelengths, got {tau.shape[1]}"
#    # Verify that computed values fall within expected physical ranges
#    assert torch.all((rho >= 0) & (rho <= 1)), "Reflectance out of bounds"
#    assert torch.all((tau >= 0) & (tau <= 1)), "Transmittance out of bounds"
