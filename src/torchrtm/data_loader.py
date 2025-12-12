"""
torchrtm.data_loader
--------------------

Data loading utilities for torchRTM models, including PROSPECT coefficient matrices,
soil reflectance spectra, and SMAC sensor coefficients.

All data is loaded as CPU-based tensors by default to ensure compatibility with
most processing environments. For GPU-based workflows, you must manually specify
the target device using the device argument, which passes the device parameter
to torch.tensor().

Example usage:

    import torch
    from torchrtm.data_loader import load_coefmat, load_soil_spectra, load_smac_sensor

    # Define target device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load coefficients and soil spectra, moving them to the target device
    coefmat = load_coefmat(device=device)
    soil_dry, soil_wet = load_soil_spectra(device=device)

    smac_dict = load_smac_sensor("sentinel2", device=device)

By default:
- CSV files are loaded into pandas DataFrames and then converted to torch.Tensors.
- Pickle files (e.g., sensor coefficients) are loaded as Python dictionaries and converted to tensors.
The user is responsible for any further conversions if needed.

Authors:
    - Peng Sun
    - Marco D. Visser
"""

import pandas as pd
import os
import torch
import pickle
import numpy as np
from importlib.resources import files
import torchrtm.data  # data files are in torchrtm/data/
from torchrtm.utils import to_device
from pathlib import Path

def load_coefmat():
    path = files(torchrtm.data).joinpath("CoefMat.csv")
    df = pd.read_csv(path)
    # Drop the wavelength column (index 0) to keep only optical coefficients
    return torch.tensor(df.iloc[:, 1:].values, dtype=torch.float32)

## This one needs the correct matrix!!!!! Only prospect5b now
def load_prospectd_matrix():
    path = files(torchrtm.data).joinpath("data_prospectd.csv")
    df = pd.read_csv(path)
    # Drop the wavelength column (index 0) to keep only optical coefficients
    return torch.tensor(df.iloc[:, 1:].values, dtype=torch.float32)
def load_prospectpro_matrix():
    path = files(torchrtm.data).joinpath("data_prospectpro.csv")
    df = pd.read_csv(path)
    # Drop the wavelength column (index 0) to keep only optical coefficients
    return torch.tensor(df.iloc[:, 1:].values, dtype=torch.float32)
def load_soil_spectra(device="cpu"):
    """
    Load and return dry and wet soil spectra as torch.Tensors.

    Args:
        device (str): Device to load the tensors on (default 'cpu').

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (dry_soil, wet_soil)
    """
    path = files("torchrtm.data").joinpath("rtm_soil.csv")
    df = pd.read_csv(path, index_col=0)
    dry = torch.tensor(df["drySoil"].values, dtype=torch.float32, device=device)
    wet = torch.tensor(df["wetSoil"].values, dtype=torch.float32, device=device)
    return dry, wet



def load_smac_sensor(sensor_name: str, device='cpu'):
    """
    Loads SMAC sensor coefficients from the given sensor name.
    
    Args:
        sensor_name (str): Name of the sensor whose coefficients need to be loaded.
        device (str): The device to load the tensors to (default is 'cpu').

    Returns:
        dict: A dictionary of SMAC coefficients loaded onto the specified device.
    """
    print(f"Attempting to load sensor: {sensor_name}")  # Debugging line
    folder = Path(__file__).parent / "data" / "sensor_information"
    file_path = folder / f"{sensor_name}.pkl"

    # Ensure the file exists
    if not file_path.exists():
        available_files = sorted(file.stem for file in folder.glob("*.pkl"))
        raise FileNotFoundError(
            f"Sensor file '{sensor_name}.pkl' not found.\n"
            f"Available files: {available_files}"
        )

    # Load the sensor file
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    # Extract coefficients from the loaded data
    coefs = data.get("SMAC_coef")
    if coefs is None:
        raise KeyError(f"Missing 'SMAC_coef' in the file {sensor_name}.pkl")

    # Convert NumPy arrays to PyTorch tensors and move them to the desired device
    return {k: to_device(torch.tensor(v, dtype=torch.float32), device) for k, v in coefs.items()}

def load_smac_sensor(sensor_name: str, device='cpu'):
    """
    Loads SMAC sensor coefficients from the given sensor name.
    
    Args:
        sensor_name (str): Name of the sensor whose coefficients need to be loaded.
        device (str): The device to load the tensors to (default is 'cpu').

    Returns:
        dict: A dictionary of SMAC coefficients loaded onto the specified device.
    """
    print(f"Attempting to load sensor: {sensor_name}")  # Debugging line
    folder = Path(__file__).parent / "data" / "sensor_information"
    file_path = folder / f"{sensor_name}.pkl"

    # Ensure the file exists
    if not file_path.exists():
        available_files = sorted(file.stem for file in folder.glob("*.pkl"))
        raise FileNotFoundError(
            f"Sensor file '{sensor_name}.pkl' not found.\n"
            f"Available files: {available_files}"
        )

    # Load the sensor file
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    # Extract coefficients from the loaded data
    coefs = data.get("SMAC_coef")
    if coefs is None:
        raise KeyError(f"Missing 'SMAC_coef' in the file {sensor_name}.pkl")
    sm_wl_initial = data.get("wl_smac")
    # Convert NumPy arrays to PyTorch tensors and move them to the desired device
    return {k: to_device(torch.tensor(v, dtype=torch.float32), device) for k, v in coefs.items()}, sm_wl_initial[:,0].astype(np.float32)