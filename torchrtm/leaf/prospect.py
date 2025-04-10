"""
torchrtm.leaf.prospect
----------------------

Implements PROSPECT-5B and PROSPECT-D models for simulating leaf reflectance and transmittance.

Models:
    - PROSPECT 5B
    - PROSPECT D
"""

import torch
from torchrtm.leaf.fresnel import tav
from torchrtm.utils import exp1, to_device
from torchrtm.data_loader import load_coefmat, load_prospectd_matrix

def plate_torch(trans, r12, t12, t21, r21, xp, yp, N, print_both=False):
    """
    Plate model to compute multiple scattering in leaf layers.

    Args:
        trans (torch.Tensor): Leaf single-layer transmittance.
        r12 (torch.Tensor): Reflectance coefficient from air to leaf.
        t12 (torch.Tensor): Transmittance from air to leaf.
        t21 (torch.Tensor): Transmittance from leaf to air.
        r21 (torch.Tensor): Reflectance from leaf to air.
        xp (torch.Tensor): Fresnel coefficient scaling parameter.
        yp (torch.Tensor): Fresnel offset parameter.
        N (torch.Tensor): Leaf structure coefficient (number of layers).
        print_both (bool): If True, returns both reflectance and transmittance.

    Returns:
        torch.Tensor or Tuple[torch.Tensor, torch.Tensor]: Reflectance (and transmittance if print_both).
    """
    trans2 = trans * trans
    ra = r12 + (t12 * t21 * r21 * trans2) / (1.0 - (r21 * r21) * trans2)
    ta = (t12 * t21 * trans) / (1.0 - (r21 * r21) * trans2)
    r90 = (ra - yp) / xp
    t90 = ta / xp
    r902 = r90 * r90
    t902 = t90 * t90

    delta = torch.sqrt((torch.pow((t902 - r902) - 1, 2) - 4.0 * r902))
    beta = (1.0 + r902 - t902 - delta) / (2.0 * r90)
    va = (1.0 + r902 - t902 + delta) / (2.0 * r90)
    beta_r90 = torch.maximum(beta - r90, torch.tensor(0.0005, device=trans.device))

    vb = torch.sqrt(beta * (va - r90) / (va * beta_r90))
    vbNN = torch.pow(vb, N.unsqueeze(1) - 1.0)
    vbNNinv = 1.0 / vbNN
    vainv = 1.0 / va
    s1 = ta * t90 * (vbNN - vbNNinv)
    s2 = ta * (va - vainv)
    s3 = va * vbNN - vainv * vbNNinv - r90 * (vbNN - vbNNinv)

    rho = ra + s1 / s3
    if not print_both:
        return rho
    tau = s2 / s3
    return rho, tau


def prospect5b(traits, N, CoefMat=None, alpha=40.0, print_both=True, device='cpu'):
    """
    PROSPECT 5B model.

    Args:
        traits (torch.Tensor): Trait matrix (e.g., [Cab, Car, Cbr, Cw, Cm]).
        N (torch.Tensor): Leaf structure parameter.
        CoefMat (torch.Tensor): Coefficient matrix with shape (wavelengths, 5); first column = n.
                                 If None, the default is loaded from package data.
        alpha (float): Incident light angle in degrees. Default is 40.
        print_both (bool): If True, returns both reflectance and transmittance.
        device (str): Torch device to use.

    Returns:
        torch.Tensor or tuple: Reflectance or (reflectance, transmittance).
    """
    if CoefMat is None:
        CoefMat = load_coefmat()  # already torch.Tensor with shape [2101, 5/6]
        
    
    n = to_device(CoefMat[:, 1], device) ## first is WL, second is n
    th_CoefMat = to_device(CoefMat[:, 2:], device) ## rest are absorbtion coefs
    alpha = to_device(alpha, device)

    t12 = tav(alpha, n)
    tav90n = tav(to_device(90.0, device), n)
    t21 = tav90n / n**2
    r12 = 1 - t12
    r21 = 1 - t21
    xp = t12 / tav90n
    yp = xp * (tav90n - 1) + 1 - t12

    
    # Make sure traits is batched.
    if traits.ndim == 1:
        traits = traits.unsqueeze(0)
        N = N.unsqueeze(0)

 
    alpha_vals = (traits.T / N).T
    k_pd = torch.matmul(alpha_vals, th_CoefMat.T)
    trans_rtm = (1 - k_pd) * torch.exp(-k_pd) + k_pd**2 * exp1(k_pd)

  #  import pdb; pdb.set_trace() ## debugger
    
    return plate_torch(trans_rtm, r12, t12, t21, r21, xp, yp, N, print_both)


def prospectd(traits, N, CoefMat=None, alpha=40.0, print_both=True, device='cpu'):
    """
    PROSPECT D model.

    Args:
        traits (torch.Tensor): Trait matrix (e.g., [Cab, Car, Anth, Cw, Cm, Cbrown, etc.]).
        N (torch.Tensor): Leaf structure parameter.
        CoefMat (torch.Tensor): Coefficient matrix with shape (wavelengths, n); first column = n.
                                 If None, the default is loaded from package data.
        alpha (float): Incident light angle in degrees. Default is 40.
        print_both (bool): If True, returns both reflectance and transmittance.
        device (str): Torch device to use.

    Returns:
        torch.Tensor or tuple: Reflectance or (reflectance, transmittance).
    """
    if CoefMat is None:
        CoefMat = load_prospectd_matrix()  # tensor of shape [2101, 7] => [n, Cab, Car, Canth, Cbrown, Cw, Cm]
        
    n = to_device(CoefMat[:, 1], device)
    th_CoefMat = to_device(CoefMat[:, 2:], device)
    alpha = to_device(alpha, device)

    t12 = tav(alpha, n)
    tav90n = tav(to_device(90.0, device), n)
    t21 = tav90n / n**2
    r12 = 1 - t12
    r21 = 1 - t21
    xp = t12 / tav90n
    yp = xp * (tav90n - 1) + 1 - t12

    if traits.ndim == 1:
        traits = traits.unsqueeze(0)
        N = N.unsqueeze(0)

    alpha_vals = (traits.T / N).T
    k_pd = torch.matmul(alpha_vals, th_CoefMat.T)
    trans_rtm = (1 - k_pd) * torch.exp(-k_pd) + k_pd**2 * exp1(k_pd)

    return plate_torch(trans_rtm, r12, t12, t21, r21, xp, yp, N, print_both)
