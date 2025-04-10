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


def dcum(a, b, theta, tol=1e-6):
    """
    Cumulative Verhoef LIDF integral using an iterative solver.
    Supports both scalar and batched inputs for a and b.

    Args:
        a (float or tensor): Shape parameter 'a'.
        b (float or tensor): Shape parameter 'b'.
        theta (torch.Tensor): Vector of angles (in degrees).
        tol (float): Tolerance for convergence (default 1e-6).

    Returns:
        torch.Tensor: A tensor of cumulative values for each angle in theta.
            If a and b are scalar: shape (13,)
            If a and b are batched: shape (B, 13)
    """
    pi = 3.14159265

    if isinstance(a, torch.Tensor) and a.ndim > 0:
        B = a.shape[0]
        result = torch.zeros(B, len(theta), dtype=torch.float32)
        for i in range(B):
            result[i] = dcum(a[i].item(), b[i].item(), theta, tol)
        return result

    # Scalar case
    cum_vals = []
    for angle in theta:
        rad = angle.item() * (pi / 180.0)
        if a > 1:
            cum = 1 - math.cos(rad)
        else:
            x = 2.0 * rad
            dx = 1.0
            iter_count = 0
            while dx > tol:
                y = a * math.sin(x) + 0.5 * b * math.sin(2.0 * x)
                dx = 0.5 * (y - x + 2.0 * rad)
                x += dx
                dx = abs(dx)
                iter_count += 1
                if iter_count > 1000:
                    break
            cum = 2.0 * (y + rad) / pi
        cum_vals.append(cum)
    return torch.tensor(cum_vals, dtype=torch.float32)


def lidf_1(na, a=-0.35, b=-0.15):
    """
    Empirical elliptical LIDF (Verhoef 1984).

    Args:
        na (torch.Tensor): Angle bin indices (13).
        a, b (float or tensor): LIDF parameters.

    Returns:
        torch.Tensor: Normalized frequency distribution.
    """
    tt = torch.tensor(
        [10, 20, 30, 40, 50, 60, 70, 80, 82, 84, 86, 88, 90],
        dtype=torch.float32, device=na.device
    )

    if isinstance(a, float) or (isinstance(a, torch.Tensor) and a.ndim == 0):
        freq = torch.zeros(len(na), dtype=torch.float32, device=na.device)
        freq[:-1] = dcum(a, b, tt[:-1])
        freq[12] = 1
#        freq[1:13] -= freq[0:12]
        freq[1:13] -= freq[0:12].clone()

    else:
        freq = torch.zeros(len(a), len(na), dtype=torch.float32, device=na.device)
        dc = dcum(a, b, tt[:-1])
        freq[:, :-1] = dc
        freq[:, 12] = 1
        freq[:, 1:13] -= dc

    return freq

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
