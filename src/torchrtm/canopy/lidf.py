"""
torchrtm.canopy.lidf
------------------

Implements Leaf Angle Distribution Functions (LIDFs) used in canopy radiative transfer models.

Functions:
    - lidf_1: Empirical Verhoef distribution
    - lidf_2: Spherical distribution based on mean ALA
    - lidf_3: Beta-distribution based LIDF (as implemented in R-package ccrtm)
    - lidf_4: Power-law LIDF (as implemented in R-package ccrtm)
"""

import torch
import math
import numpy as np
from scipy.special import betainc
from torchrtm.utils import to_device
import torch._dynamo
import math

from torch import (
    sin, cos, tan, acos, asin, log, pow, sqrt, square, exp
)

from torchrtm.utils import to_device


import numpy as np

def dcum_numpy(a, b, t):
    """
    NumPy implementation of cumulative LIDF distribution.
    For single a, b, t values.
    """
    rd = np.pi / 180
    eps = 1e-8
    if a > 1:
        return 1 - rd * t

    x = 2 * rd * t
    p = x
    mask = False
    while not mask:
        y = a * np.sin(x) + 0.5 * b * np.sin(2 * x)
        dx = 0.5 * (y - x + p)
        x += dx
        if abs(dx) <= eps:
            mask = True
    return (2 * y + p) / np.pi

def lidf_1(na, a=-0.3500, b=-0.1500):
    """
    Calculate the Leaf Inclination Distribution Function (LIDF) based on parameters 'a' and 'b'.

    Args:
        na (tensor): Input data (often the wavelength values).
        a (float): LIDF parameter a (default is -0.3500).
        b (float): LIDF parameter b (default is -0.1500).

    Returns:
        tensor: The LIDF values for the given parameters.
    """
    device=na.device
    tt = (np.concatenate([(np.array(range(8)) + 1) * 10, 80 + (np.array([9, 10, 11, 12]) - 8) * 2]))  # Wavelength grid
    if len(a) == 1:
        na, a, b = na.data.cpu().numpy(), a.data.cpu().numpy(), b.data.cpu().numpy()
        #freq = np.zeros((na))
        freq = np.zeros(len(na))
        freq[:-1] = list(map(lambda t: dcum_numpy(a, b, t), (tt)))
        freq[12] = 1
        freq[1:13] = freq[1:13] - freq[0:12]

        return torch.tensor(freq).to(device)
    else:

        freq = torch.zeros(len(a), len(na)).to(device)
        if not torch.is_tensor(a):
            a = torch.tensor([a], device=device)
        if not torch.is_tensor(b):
            b = torch.tensor([b], device=device)
        if not torch.is_tensor(tt):
            tt = torch.tensor([tt], device=device)
        freq[:, :-1] = dcum_vectorized(a, b, tt, device = a.device if isinstance(a, torch.Tensor) else 'cpu')
        freq[:, 12] = 1
        freq[:, 1:13] = freq[:, 1:13] - freq[:, 0:12]
        return freq

    

def dcum_vectorized(a, b, theta_deg,device, max_iter=30, tol=1e-6):
    """
    GPU vectorized version of the Verhoef cumulative LIDF integral approximation.
    Args:
        a, b: tensors of shape [B] or scalars
        theta_deg: tensor of shape [T], angles in degrees (e.g., 13 angles)
    Returns:
        Tensor of shape [B, T]
    """

    pi = torch.tensor(math.pi, device=device)

    # Prepare tensors
    
    B = a.shape[0]
    T = len(theta_deg)

    theta_rad = theta_deg.to(device) * pi / 180.0  # [T]

    # Broadcast shapes: [B, T]
    a = a.view(-1, 1)              # [B, 1]
    b = b.view(-1, 1)              # [B, 1]
    theta = theta_rad.view(1, -1)  # [1, T]

    #x = 2 * theta.clone().expand(B, T)  # initial guess
    x = 2 * theta.clone().expand(B, -1)  # initial guess

    for _ in range(max_iter):
        sinx = torch.sin(x)
        sin2x = torch.sin(2 * x)
        y = a * sinx + 0.5 * b * sin2x
        dx = 0.5 * (y - x + 2 * theta)
        x = x + dx
        if torch.max(torch.abs(dx)) < tol:
            break

    cum = 2.0 * (y + theta) / pi
    return cum

def lidf_2(na, ala):
    """
    Spherical LIDF based on mean leaf angle (ALA).

    Args:
        na (torch.Tensor): Angle bin indices.
        ala (torch.Tensor): Mean leaf inclination angle.

    Returns:
        torch.Tensor: LIDF frequency per angle.
    """
    pi = torch.tensor(np.pi, device=ala.device)
    tx2 = torch.tensor([0, 10, 20, 30, 40, 50, 60, 70, 80, 82, 84, 86, 88], device=ala.device)
    tx1 = torch.tensor([10, 20, 30, 40, 50, 60, 70, 80, 82, 84, 86, 88, 90], device=ala.device)
    tl1 = tx1 * pi / 180
    tl2 = tx2 * pi / 180
    excent = torch.exp(-1.6184e-5 * ala ** 3 + 2.1145e-3 * ala ** 2 - 1.2390e-1 * ala + 3.2491)

    if ala.ndim > 0:
        excent = excent.expand(len(na), len(ala)).T
        x1 = excent / torch.sqrt(1 + excent ** 2 * torch.tan(tl1.expand(len(ala), len(na))) ** 2)
        x2 = excent / torch.sqrt(1 + excent ** 2 * torch.tan(tl2.expand(len(ala), len(na))) ** 2)
        alpha = excent / torch.sqrt(torch.abs(1 - excent ** 2))
        alpha2 = alpha ** 2
        freq = torch.zeros([len(ala), 13], device=ala.device)

        mask = excent > 1
        if mask.any():
            alpx1 = torch.sqrt(alpha2[mask] + x1[mask] ** 2)
            alpx2 = torch.sqrt(alpha2[mask] + x2[mask] ** 2)
            freq[mask] = torch.abs(
                x1[mask] * alpx1 + alpha2[mask] * torch.log(x1[mask] + alpx1)
                - x2[mask] * alpx2 - alpha2[mask] * torch.log(x2[mask] + alpx2)
            )

        mask = excent < 1
        if mask.any():
            almx1 = torch.sqrt(alpha2[mask] - x1[mask] ** 2)
            almx2 = torch.sqrt(alpha2[mask] - x2[mask] ** 2)
            freq[mask] = torch.abs(
                x1[mask] * almx1 + alpha2[mask] * torch.asin(x1[mask] / alpha[mask])
                - x2[mask] * almx2 - alpha2[mask] * torch.asin(x2[mask] / alpha[mask])
            )

        finalfreq = freq / torch.sum(freq, dim=1, keepdim=True)
    else:
        x1 = excent / torch.sqrt(1 + excent ** 2 * torch.tan(tl1) ** 2)
        x2 = excent / torch.sqrt(1 + excent ** 2 * torch.tan(tl2) ** 2)
        alpha = excent / torch.sqrt(torch.abs(1 - excent ** 2))
        alpha2 = alpha ** 2

        if excent > 1:
            alpx1 = torch.sqrt(alpha2 + x1 ** 2)
            alpx2 = torch.sqrt(alpha2 + x2 ** 2)
            freq = torch.abs(x1 * alpx1 + alpha2 * torch.log(x1 + alpx1)
                             - x2 * alpx2 - alpha2 * torch.log(x2 + alpx2))
        else:
            almx1 = torch.sqrt(alpha2 - x1 ** 2)
            almx2 = torch.sqrt(alpha2 - x2 ** 2)
            freq = torch.abs(x1 * almx1 + alpha2 * torch.asin(x1 / alpha)
                             - x2 * almx2 - alpha2 * torch.asin(x2 / alpha))

        finalfreq = freq / torch.sum(freq)

    return finalfreq.to(dtype=torch.float32)


def lidf_3(na, a, b):
    """
    LIDF from Beta distribution.

    Args:
        na (torch.Tensor): Angle bins (13).
        a (float or tensor): Shape parameter.
        b (float or tensor): Shape parameter.

    Returns:
        torch.Tensor: LIDF frequency values.
    """
    t1 = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 82, 84, 86, 88])
    t2 = np.array([10, 20, 30, 40, 50, 60, 70, 80, 82, 84, 86, 88, 90])

    a = a.detach().cpu().numpy()
    b = b.detach().cpu().numpy()

    if a.ndim == 0:
        freq = betainc(a, b, t2 / 90) - betainc(a, b, t1 / 90)
    else:
        freq = np.zeros((len(a), len(na)))
        for i in range(len(a)):
            freq[i] = betainc(a[i], b[i], t2 / 90) - betainc(a[i], b[i], t1 / 90)

    return torch.tensor(freq, dtype=torch.float32, device=na.device)


def lidf_4(na, theta):
    """
    Power-law LIDF.

    Args:
        na (torch.Tensor): Angle bins.
        theta (float or tensor): Exponent shape parameter.

    Returns:
        torch.Tensor: LIDF frequency distribution.
    """
    t1 = torch.tensor([0, 10, 20, 30, 40, 50, 60, 70, 80, 82, 84, 86, 88], device=na.device)
    t2 = torch.tensor([10, 20, 30, 40, 50, 60, 70, 80, 82, 84, 86, 88, 90], device=na.device)

    if theta.ndim == 0:
        freq = (t2 / 90) ** theta - (t1 / 90) ** theta
    else:
        theta = theta.unsqueeze(1).expand(-1, len(na))
        in2 = (t2.unsqueeze(0) / 90) ** theta
        in1 = (t1.unsqueeze(0) / 90) ** theta
        freq = in2 - in1

    return freq.to(torch.float32)
