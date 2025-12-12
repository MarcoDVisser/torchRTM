"""
torchrtm.leaf.fresnel
---------------------

Fresnel functions used in PROSPECT models (e.g., tav).
Implements analytical approximation from Stokes (1862) and Allen (1969).
"""

import torch

def tav(theta, n):
    """
    Fresnel average transmittance calculation for PROSPECT models.

    Args:
        theta (float or torch.Tensor): Incident angle in degrees.
        n (torch.Tensor): Refractive index (shape: [2101,]).

    Returns:
        torch.Tensor: Average transmittance (same shape as n, or [B, 2101] if batched).
    """
    # Ensure theta and n are tensors on the same device
    device = n.device
    if not isinstance(theta, torch.Tensor):
        theta = torch.tensor(theta, device=device)
    if not isinstance(n, torch.Tensor):
        n = torch.tensor(n, device=device)

    pi = torch.tensor(torch.pi, device=device)

    # Handle edge cases and determine batching
    try:
        len_theta = len(theta)
        if torch.all(theta == theta[0]):
            theta = theta[0]
            if theta == 0:
                return 4.0 * n / ((n + 1.0) ** 2)
            use_batch = False
        else:
            n = n.expand(len_theta, 2101)
            use_batch = True
    except TypeError:
        use_batch = False
        if theta == 0:
            return 4.0 * n / ((n + 1.0) ** 2)

    # Begin computation
    n2 = n ** 2
    np = n2 + 1.0
    nm = n2 - 1.0
    a = ((n + 1.0) ** 2) / 2.0
    k = -((n2 - 1.0) ** 2) / 4.0
    ds = torch.sin(theta * pi / 180.0)
    ds2 = ds ** 2
    k2 = k ** 2
    nm2 = nm ** 2

    if not use_batch:
        if theta != 90:
            b1 = torch.sqrt((ds2 - np / 2.0) ** 2 + k)
        else:
            b1 = torch.tensor(0.0, device=device)
    else:
        ds2 = ds2.unsqueeze(1).expand(len_theta, 2101)
        np = np.expand(len_theta, 2101)
        k = k.expand(len_theta, 2101)
        b1 = torch.sqrt((ds2 - np / 2.0) ** 2 + k)

    b2 = ds2 - np / 2.0
    b = b1 - b2

    ts = (k2 / (6.0 * b ** 3) + k / b - b / 2.0) - (k2 / (6.0 * a ** 3) + k / a - a / 2.0)
    tp1 = -2.0 * n2 * (b - a) / (np ** 2)
    tp2 = -2.0 * n2 * np * torch.log(b / a) / nm2
    tp3 = n2 * (1.0 / b - 1.0 / a) / 2.0
    n22 = n2 ** 2
    np3 = np ** 3
    tp4 = 16.0 * n22 * (n22 + 1.0) * torch.log((2.0 * np * b - nm2) / (2.0 * np * a - nm2)) / (np3 * nm2)
    tp5 = 16.0 * n2 ** 3 * (1.0 / (2.0 * np * b - nm2) - 1.0 / (2.0 * np * a - nm2)) / np3
    tp = tp1 + tp2 + tp3 + tp4 + tp5

    f = (ts + tp) / (2.0 * ds2)

    if use_batch and torch.any(theta == 0):
        f[theta == 0] = 4.0 * n[theta == 0] / ((n[theta == 0] + 1.0) ** 2)

    return f
