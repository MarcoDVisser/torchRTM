"""
torchrtm.canopy.sail_radiative
------------------------------

Radiative transfer calculations for canopy reflectance using the SAIL model.
Computes canopy reflectance and transmittance using SAIL two-stream approximation.
Includes auxiliary J-functions for radiative transfer integrals in SAIL.
"""

import torch
from torch import exp

@torch.jit.script
def RTgeom(rho, tau, ddb, ddf, sdb, sdf, dob, dof, sob, sof):
    """
    Computes single and multiple scattering terms for the canopy.

    Args:
        rho (tensor): Leaf reflectance.
        tau (tensor): Leaf transmittance.
        ddb, ddf, sdb, sdf (tensor): Directional terms.
        dob, dof, sob, sof (tensor): Observer and solar backscatter terms.

    Returns:
        list: [sigb, att, m, sb, sf, vb, vf, w]
    """
    sigb = ddb * rho + ddf * tau
    sigf = ddf * rho + ddb * tau
    att = 1 - sigf
    m2 = (att + sigb) * (att - sigb)
    m2[m2 < 0] = 0
    m = torch.sqrt(m2)

    sb = sdb * rho + sdf * tau
    sf = sdf * rho + sdb * tau
    vb = dob * rho + dof * tau
    vf = dof * rho + dob * tau
    w = sob * rho + sof * tau

    return [sigb, att, m, sb, sf, vb, vf, w]



from typing import Tuple

@torch.jit.script
def Jfunc1_cached(k: torch.Tensor, l: torch.Tensor, t: torch.Tensor,
                  kt: torch.Tensor, lt: torch.Tensor) -> torch.Tensor:
    del_i = (k - l) * t
    exp_k = torch.exp(-kt)
    exp_l = torch.exp(-lt)
    approx = 0.5 * t * (exp_k + exp_l) * (1 - del_i * del_i / 12.0)
    cond = torch.abs(del_i) <= 1e-3
    eps = 1e-12
    denom = k - l
    denom = torch.where(torch.abs(denom) < eps, torch.full_like(denom, eps), denom)
    exact = (exp_l - exp_k) / denom
    Jout = torch.where(cond, approx, exact)
    return Jout
from typing import List

@torch.jit.script
def sail_BDRF(
    w: torch.Tensor,
    lai: torch.Tensor,
    sumint: torch.Tensor,
    tsstoo: torch.Tensor,
    rsoil: torch.Tensor,
    rdd: torch.Tensor,
    tdd: torch.Tensor,
    tsd: torch.Tensor,
    rsd: torch.Tensor,
    tdo: torch.Tensor,
    rdo: torch.Tensor,
    tss: torch.Tensor,
    too: torch.Tensor,
    rsod: torch.Tensor,
    len_batch: int  # 👈 这里显式声明类型
) -> List[torch.Tensor]:
    """
    Final bidirectional reflectance factor (BRDF) computation.

    Returns:
        list: [rddt, rsdt, rdot, rsot, tsd, tdd, rdd]
    """
    rsos = w * lai * sumint  # Single scattering term
    rso = rsos + rsod
    dn = 1 - rsoil * rdd
    dn[dn == 0] = 1e-8  # numerical safeguard

    rddt = rdd + tdd * rsoil * tdd / dn
    rsdt = rsd + (tsd + tss) * rsoil * tdd / dn
    rdot = rdo + tdd * rsoil * (tdo + too) / dn
    rsodt = rsod + ((tss + tsd) * tdo + (tsd + tss * rsoil * rdd) * too) * rsoil / dn
    rsost = rsos + tsstoo * rsoil
    rsot = rsost + rsodt

    return [rddt, rsdt, rdot, rsot, tsd, tdd, rdd]

@torch.jit.script
def Jfunc1(k, l, t):
    del_i = (k - l) * t
    exp_k = torch.exp(-k * t)
    exp_l = torch.exp(-l * t)
    approx = 0.5 * t * (exp_k + exp_l) * (1 - del_i ** 2 / 12)
    exact = (exp_l - exp_k) / (k - l)
    mask = torch.abs(del_i) <= 1e-3
    Jout = torch.where(mask, approx, exact)
    return Jout

@torch.jit.script
def Jfunc2(k, l, t):
    return (1 - exp(-(k + l) * t)) / (k + l)

@torch.jit.script
def Jfunc3(k, l, t):
    return (1 - exp(-(k + l) * t)) / (k + l)

@torch.jit.script
def Jfunc4(m, t):
    del_i = m * t
    out = torch.zeros_like(del_i)
    mask1 = del_i > 1e-3
    mask2 = del_i <= 1e-3
    e = exp(-del_i)
    out[mask1] = (1 - e[mask1]) / (m[mask1] * (1 + e[mask1]))
    out[mask2] = 0.5 * t[mask2] * (1 - del_i[mask2] ** 2 / 12)
    return out

@torch.jit.script
def ReflTrans(rho, tau, lai, att, m, sigb, ks, ko, sf, sb, vf, vb): #, use_batch=True
    mlai = m * lai
    e1 = torch.exp(-mlai)
    e2 = e1 ** 2

    rinf = (att - m) / sigb
    rinf2 = rinf ** 2
    re = rinf * e1
    denom = 1 - rinf2 * e2

    kslai = ks * lai
    kolai = ko * lai
    tss = torch.exp(-kslai)
    too = torch.exp(-kolai)

    J1ks = Jfunc1_cached(ks, m, lai, kslai, mlai)
    J2ks = Jfunc2(ks, m, lai)
    J1ko = Jfunc1_cached(ko, m, lai, kolai, mlai)
    J2ko = Jfunc2(ko, m, lai)

    Ps = (sf + sb * rinf) * J1ks
    Qs = (sf * rinf + sb) * J2ks
    Pv = (vf + vb * rinf) * J1ko
    Qv = (vf * rinf + vb) * J2ko

    rdd = rinf * (1 - e2) / denom
    tdd = (1 - rinf2) * e1 / denom
    tsd = (Ps - re * Qs) / denom
    rsd = (Qs - re * Ps) / denom
    tdo = (Pv - re * Qv) / denom
    rdo = (Qv - re * Pv) / denom

    z = Jfunc3(ks, ko, lai)
    g1 = (z - J1ks * too) / (ko + m)
    g2 = (z - J1ko * tss) / (ks + m)
    Tv1 = (vf * rinf + vb) * g1
    Tv2 = (vf + vb * rinf) * g2
    T1 = Tv1 * (sf + sb * rinf)
    T2 = Tv2 * (sf * rinf + sb)
    T3 = (rdo * Qs + tdo * Ps) * rinf

    rsod = (T1 + T2 - T3) / (1 - rinf2)

    #if use_batch:
    return [rdd, tdd, tsd, rsd, tdo, rdo, tss, too, rsod]
    #else:
    #    
    #return [rdd.flatten(), tdd.flatten(), tsd.flatten(), rsd.flatten(),
    #            tdo.flatten(), rdo.flatten(), tss, too, rsod.flatten()]