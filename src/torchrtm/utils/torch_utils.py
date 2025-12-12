"""
torchrtm.utils.torch_utils
--------------------------

General-purpose torch utilities.
"""

import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pkg_resources  

def to_device(x, device='cpu'):
    """
    Moves a tensor (or other data structure) to the specified device.
    Supports tensors, lists, tuples, dicts, and scalar floats.
    """
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, float):  # Handle float type by converting it to a tensor
        return torch.tensor(x, device=device, dtype=torch.float32)
    elif isinstance(x, np.ndarray):  # Handle numpy ndarray by converting it to a tensor
        return torch.tensor(x, device=device, dtype=torch.float32)
    elif isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    elif isinstance(x, (list, tuple)):
        return type(x)(to_device(v, device) for v in x)
    else:
        raise TypeError(f"Unsupported type for to_device: {type(x)}")

## close to original by Peng
#def to_device(x, device='cpu'):
#    """Moves the tensor to the specified device, ensuring compatibility with CPU or GPU."""
#    return torch.tensor(x, dtype=torch.float32).to(device)


def is_batch(tensor):
    """
    Check if tensor is batched.

    Returns:
        bool: True if tensor has 2+ dimensions.
    """
    return tensor.ndim >= 2

class SpectraDataset(Dataset):
    def __init__(self, simulated_spectra, initial_traits):
        self.simulated_spectra = torch.tensor(simulated_spectra, dtype=torch.float32)
        self.initial_traits = torch.tensor(initial_traits, dtype=torch.float32)

    def __len__(self):
        return len(self.simulated_spectra)

    def __getitem__(self, idx):
        return self.simulated_spectra[idx], self.initial_traits[idx]
        # Assuming simulated_spectra and initial_traits are numpy arrays




def get_file_path(filename):
    return pkg_resources.resource_filename('torchrtm', f'data/{filename}')


def load_prosail_params(paras: pd.DataFrame = None):

    if paras is not None:
        if isinstance(paras, pd.DataFrame):
            return paras
        else:
            raise TypeError("paras must be a pandas DataFrame or None.")
    else:
        file_path = get_file_path('all_para.csv')
        return pd.read_csv(file_path)

def get_default_range():
    return load_prosail_params()



# Define a function to get max and min values based on param_type
def get_param_ranges(param_type,para_addr=None):

    if para_addr is None:
        paras = load_prosail_params()
    elif isinstance(para_addr, pd.DataFrame):
        paras = para_addr
    elif isinstance(para_addr, str):
        paras = pd.read_csv(para_addr)
    else:
        raise TypeError("para_addr must be a CSV file path, a pandas DataFrame, or None.")

    """
    Get the max and min values for the specified parameter type.

    Parameters:
        param_type (str): Type of parameters ('atom', 'prosail', or 'prospect').

    Returns:
        tuple: (x_max, x_min) for the selected parameter type.
    """
    if param_type == 'atom':
        # Example: atomic parameters use the first 6 rows (you can adjust this based on your dataset)
        x_max = paras.iloc[-3:, :]['max'].values
        x_min = paras.iloc[-3:, :]['min'].values
    elif param_type == 'prosail':
        # Example: PROSAIL parameters use the first 15 rows
        x_max = paras.iloc[:17, :]['max'].values
        x_min = paras.iloc[:17, :]['min'].values
    elif model == 'prospectd' ormodel == 'prospect5d'  :
        # Example: Prospect parameters use a specific range, adjust as needed
        x_max = paras.iloc[8:15, :]['max'].values
        x_min = paras.iloc[8:15, :]['min'].values
    elif model == 'prospect5b' or model == 'prospect5':
        # Example: Prospect parameters use a specific range, adjust as needed
        x_max = paras.iloc[8:14, :]['max'].values
        x_min = paras.iloc[8:14, :]['min'].values
    elif param_type == 'prospectpro':
        # Example: Prospect parameters use a specific range, adjust as needed
        x_max = paras.iloc[[8,9,10,11,12,14,15,16], :]['max'].values
        x_min = paras.iloc[[8,9,10,11,12,14,15,16], :]['min'].values
    else:
        raise ValueError(f"Unknown param_type: {param_type}, select one of these: atom, prosail, prospectd, prospect5b")
    
    return x_max, x_min

# Unified normalization and denormalization function for both torch and numpy
def normalize_parameters(x, param_type='prosail', fitting=True, use_torch=True,para_addr=None):
    """
    Normalize or denormalize the input parameters using either PyTorch or NumPy.
    
    Parameters:
        x (torch.Tensor or np.ndarray): Input parameter array.
        param_type (str): Type of parameters ('atom', 'prosail', or 'prospect').
        fitting (bool): Whether to normalize (True) or denormalize (False).
        use_torch (bool): If True, uses PyTorch. If False, uses NumPy.
    
    Returns:
        torch.Tensor or np.ndarray: Normalized or denormalized parameters.
    """
    # Get the correct ranges based on param_type
    x_max, x_min = get_param_ranges(param_type,para_addr=para_addr)
    
    if use_torch:
        x_max = torch.tensor(x_max, dtype=torch.float32)
        x_min = torch.tensor(x_min, dtype=torch.float32)

        try:
            x2 = x.clone()
        except:
            x2 = x.detach().clone() if isinstance(x, torch.Tensor) else torch.tensor(x)

        if len(x2.shape) == 2:
            for i in range(x2.shape[-1]):
                if fitting:
                    x2[:, i] = (x[:, i] - x_min[i]) / (x_max[i] - x_min[i])
                else:
                    x2[:, i] = x[:, i] * (x_max[i] - x_min[i]) + x_min[i]

    else:
        x_max = np.array(x_max)
        x_min = np.array(x_min)

        try:
            x2 = x.copy()
        except:
            x2 = np.copy(x)

        if len(x2.shape) == 2:
            for i in range(x2.shape[-1]):
                if fitting:
                    x2[:, i] = (x[:, i] - x_min[i]) / (x_max[i] - x_min[i])
                else:
                    x2[:, i] = x[:, i] * (x_max[i] - x_min[i]) + x_min[i]


    return x2