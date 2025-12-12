"""
torchrtm.canopy.hotspot
-----------------------

Implements hotspot correction for canopy bidirectional reflectance.
"""

import torch
import torch._dynamo

##@torch.compile  # 

@torch.jit.script
def HotSpot(lai, q, tss, ks, ko, dso):
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

    # 条件计算 alf
    q_pos = q > 0
    alf[q_pos] = (dso[q_pos] / q[q_pos]) * 2 / (ks[q_pos] + ko[q_pos])
    alf = torch.clamp_max(alf, 200)

    fhot = lai * torch.sqrt(ko * ks)
    ca = torch.exp(-alf)
    fint = (1 - ca) * 0.05

    sumint = torch.zeros_like(alf)
    x1 = torch.zeros_like(alf)
    y1 = torch.zeros_like(alf)
    f1 = torch.ones_like(alf)

    eps = 1e-6  # 防止除以0

    for i in range(1, 21):
        i_float = float(i)
        i_tensor = torch.full_like(alf, i_float)
        mask = i_tensor < 20
        x2 = torch.where(mask, -torch.log(1 - i_tensor * fint) / alf, torch.ones_like(alf))

        exp_alf_x2 = torch.exp(-alf * x2)
        numerator = (1 - exp_alf_x2)
        y2 = -(ko + ks) * lai * x2 + fhot * numerator / alf

        f2 = torch.exp(y2)
        denom = torch.where(torch.abs(y2 - y1) < eps, torch.full_like(y2, eps), y2 - y1)
        sumint += (f2 - f1) * (x2 - x1) / denom

        x1 = x2
        y1 = y2
        f1 = f2

    tsstoo = f1
    alf_zero = alf == 0
    tsstoo = torch.where(alf_zero, tss, tsstoo)
    sumint = torch.where(alf_zero, (1 - tss) / (ks * lai), sumint)

    return tsstoo, sumint