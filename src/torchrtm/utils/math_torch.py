"""
torchrtm.utils.math_torch
--------------------------

Math utilities that wrap SciPy functions for safe use in PyTorch tensors.
"""

import torch
from scipy.special import exp1 as scipy_exp1, betainc as scipy_betainc


def exp1(x):
    """
    Compute exponential integral E1(x) element-wise, safely on torch tensors.

    Args:
        x (torch.Tensor): Input values (must be positive).

    Returns:
        torch.Tensor: E1(x) values.
    """
    x_cpu = x.detach().cpu().numpy()
    y_cpu = scipy_exp1(x_cpu)
    return torch.tensor(y_cpu, dtype=torch.float32, device=x.device)


def betainc(a, b, x):
    """
    Regularized incomplete beta function.

    Args:
        a (torch.Tensor): Alpha shape parameter.
        b (torch.Tensor): Beta shape parameter.
        x (torch.Tensor): Evaluation point.

    Returns:
        torch.Tensor: Regularized incomplete beta function value.
    """
    a_, b_, x_ = a.detach().cpu().numpy(), b.detach().cpu().numpy(), x.detach().cpu().numpy()
    result = scipy_betainc(a_, b_, x_)
    return torch.tensor(result, dtype=torch.float32, device=x.device)



