"""
torchrtm.utils.torch_utils
--------------------------

General-purpose torch utilities.
"""

import torch
import numpy as np

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

