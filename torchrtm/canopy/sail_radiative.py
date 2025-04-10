"""
torchrtm.canopy.sail_radiative
------------------------------

Radiative transfer calculations for canopy reflectance using the SAIL model.
Computes canopy reflectance and transmittance using SAIL two-stream approximation.
Includes auxiliary J-functions for radiative transfer integrals in SAIL.
"""

import torch
from torch import exp


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


def ReflTrans(rho, tau, lai, att, m, sigb, ks, ko, sf, sb, vf, vb, use_batch=True):
    """
    Computes reflectance and transmittance of the canopy.

    Args:
        rho (tensor): Leaf reflectance.
        tau (tensor): Leaf transmittance.
        lai (tensor): Leaf Area Index.
        att (tensor): Attenuation term.
        m (tensor): Auxiliary term.
        sigb (tensor): Backscatter component.
        ks, ko (tensor): Extinction coefficients (solar and observer).
        sf, sb, vf, vb (tensor): Scattering terms.
        use_batch (bool): Enable batch support.

    Returns:
        list: [rdd, tdd, tsd, rsd, tdo, rdo, tss, too, rsod]
    """
    e1 = exp(-m * lai)
    e2 = e1 ** 2
    rinf = (att - m) / sigb
    rinf2 = rinf ** 2
    re = rinf * e1
    denom = 1 - rinf2 * e2

    J1ks = Jfunc1(ks, m, lai)
    J2ks = Jfunc2(ks, m, lai)
    J1ko = Jfunc1(ko, m, lai)
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
    tss = exp(-ks * lai)
    too = exp(-ko * lai)

    z = Jfunc3(ks, ko, lai)
    g1 = (z - J1ks * too) / (ko + m)
    g2 = (z - J1ko * tss) / (ks + m)
    Tv1 = (vf * rinf + vb) * g1
    Tv2 = (vf + vb * rinf) * g2
    T1 = Tv1 * (sf + sb * rinf)
    T2 = Tv2 * (sf * rinf + sb)
    T3 = (rdo * Qs + tdo * Ps) * rinf

    rsod = (T1 + T2 - T3) / (1 - rinf2)

    if use_batch:
        return [rdd, tdd, tsd, rsd, tdo, rdo, tss, too, rsod]
    else:
        return [rdd.flatten(), tdd.flatten(), tsd.flatten(), rsd.flatten(),
                tdo.flatten(), rdo.flatten(), tss, too, rsod.flatten()]


def sail_BDRF(w, lai, sumint, tsstoo, rsoil, rdd, tdd, tsd, rsd, tdo, rdo, tss, too, rsod, len_batch, use_batch=True):
    """
    Final bidirectional reflectance factor (BRDF) computation.

    Returns:
        list: [rddt, rsdt, rdot, rsot, tsd, tdd, rdd]
    """
    rsos = w * lai * sumint  # Single scattering term
    if not use_batch:
        rsos = rsos.flatten()

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


def Jfunc1(k, l, t):
    del_i = (k - l) * t
    Jout = (exp(-l * t) - exp(-k * t)) / (k - l)
    mask = abs(del_i) <= 1e-3
    Jout[mask] = (0.5 * t * (exp(-k * t) + exp(-l * t)) * (1 - del_i ** 2 / 12))[mask]
    return Jout


def Jfunc2(k, l, t):
    return (1 - exp(-(k + l) * t)) / (k + l)


def Jfunc3(k, l, t):
    return (1 - exp(-(k + l) * t)) / (k + l)


def Jfunc4(m, t):
    del_i = m * t
    out = torch.zeros_like(del_i)
    mask1 = del_i > 1e-3
    mask2 = del_i <= 1e-3
    e = exp(-del_i)
    out[mask1] = (1 - e[mask1]) / (m[mask1] * (1 + e[mask1]))
    out[mask2] = 0.5 * t[mask2] * (1 - del_i[mask2] ** 2 / 12)
    return out
