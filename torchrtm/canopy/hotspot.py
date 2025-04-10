"""
torchrtm.canopy.hotspot
-----------------------

Implements hotspot correction for canopy bidirectional reflectance.
"""

import torch

def HotSpot(lai, q, tss, ks, ko, dso, use_batch=True):
    """
    HotSpot correction for BRDF based on Breon formulation.

    Args:
        lai (torch.Tensor): Leaf Area Index.
        q (torch.Tensor): Hotspot parameter.
        tss (torch.Tensor): Soil transmittance.
        ks (torch.Tensor): Extinction coefficient (solar).
        ko (torch.Tensor): Extinction coefficient (observer).
        dso (torch.Tensor): Angular distance between sun and sensor.
        use_batch (bool): Whether using batched mode.

    Returns:
        tuple: (tsstoo, sumint) for use in BDRF.
    """
    device = lai.device
    alf = torch.full_like(q, 1e6, device=device)
    q_mask = q > 0
    alf[q_mask] = (dso[q_mask] / q[q_mask]) * 2 / (ks[q_mask] + ko[q_mask])
    alf = torch.clamp(alf, max=200)

    fhot = lai * torch.sqrt(ko * ks)
    ca = torch.exp(-alf)
    fint = (1 - ca) * 0.05

    sumint = torch.zeros_like(alf)
    x1 = torch.zeros_like(alf)
    y1 = torch.zeros_like(alf)
    f1 = torch.ones_like(alf)

    for i in range(1, 21):
        x2 = -torch.log(1 - i * fint) / alf if i < 20 else torch.ones_like(alf)
        y2 = -(ko + ks) * lai * x2 + fhot * (1 - torch.exp(-alf * x2)) / alf
        f2 = torch.exp(y2)
        denom = (y2 - y1).clamp(min=1e-6)
        sumint += (f2 - f1) * (x2 - x1) / denom
        x1, y1, f1 = x2, y2, f2

    tsstoo = f1
    if use_batch:
        alf_zero = alf == 0
        tsstoo = torch.where(alf_zero, tss, tsstoo)
        sumint = torch.where(alf_zero, (1 - tss) / (ks * lai), sumint)
    else:
        if alf.item() == 0:
            tsstoo = tss
            sumint = (1 - tss) / (ks * lai)

    return tsstoo, sumint
