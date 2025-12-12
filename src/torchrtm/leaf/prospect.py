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
from torchrtm.data_loader import load_coefmat, load_prospectd_matrix, load_prospectpro_matrix
from typing import Tuple

def exp1_pytorch(x):
    """
    PyTorch implementation of exponential integral E1(x)

    Uses different algorithms for different ranges to optimize accuracy:
    - Small x: Series expansion around x=0
    - Large x: Asymptotic expansion

    Args:
        x: torch.Tensor, input values (must be positive)

    Returns:
        torch.Tensor, E1(x) values
    """
    x = torch.as_tensor(x, dtype=torch.float32)
    original_shape = x.shape
    x = x.flatten()

    # Initialize result tensor
    result = torch.zeros_like(x)

    # Split computation based on x magnitude for optimal accuracy
    small_mask = x < 10.0
    large_mask = x >= 10.0

    # For small to medium x: use series expansion
    # E1(x) = -γ - ln(x) + x - x²/4 + x³/18 - x⁴/96 + ...
    if torch.any(small_mask):
        x_small = x[small_mask]
        euler_gamma = 0.5772156649015329

        # Improved series expansion with better numerical stability
        log_x = torch.log(x_small)

        # E1(x) = -γ - ln(x) + Σ((-1)^(n+1) * x^n / (n * n!))
        # Use iterative approach to avoid factorial overflow
        series = torch.zeros_like(x_small)
        term = x_small.clone()  # First term: x/1!

        for n in range(1, 50):  # Limit iterations to prevent overflow
            # Add current term with proper sign
            series += (-1)**(n+1) * term / n

            # Prepare next term: multiply by x and divide by (n+1)
            term = term * x_small / (n + 1)

            # Check convergence - if term becomes too small, stop
            if torch.max(torch.abs(term)) < 1e-16:
                break

        result[small_mask] = -euler_gamma - log_x + series

    # For large x: use asymptotic expansion
    # E1(x) ≈ e^(-x)/x * (1 - 1/x + 2!/x² - 3!/x³ + ...)
    if torch.any(large_mask):
        x_large = x[large_mask]
        exp_neg_x = torch.exp(-x_large)
        inv_x = 1.0 / x_large

        # Asymptotic series: 1 + Σ((-1)^n * n! / x^n)
        # Use iterative computation to avoid factorial overflow
        asymptotic = torch.ones_like(x_large)
        term = -inv_x  # First term: -1/x

        for n in range(1, 15):  # Limit to prevent overflow
            asymptotic += term
            # Next term: multiply by -(n+1)/x
            term = term * (-(n + 1)) * inv_x

            # Check convergence
            if torch.max(torch.abs(term)) < 1e-16:
                break

        result[large_mask] = exp_neg_x * inv_x * asymptotic

    return result.reshape(original_shape)




#@torch.jit.script
def plate_torch(trans: torch.Tensor, r12: torch.Tensor, t12: torch.Tensor,
                     t21: torch.Tensor, r21: torch.Tensor, xp: torch.Tensor,
                     yp: torch.Tensor, N: torch.Tensor,print_both= True) -> Tuple[torch.Tensor, torch.Tensor]:
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

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Reflectance and Transmittance.
    """
    trans2 = trans * trans
    r21_sq = r21 * r21
    denom = 1.0 - r21_sq * trans2

    ra = r12 + (t12 * t21 * r21 * trans2) / denom
    ta = (t12 * t21 * trans) / denom

    r90 = (ra - yp) / xp
    t90 = ta / xp

    r902 = r90 * r90
    t902 = t90 * t90

    delta = torch.sqrt(((t902 - r902 - 1)**2 - 4 * r902))
    beta = (1.0 + r902 - t902 - delta) / (2.0 * r90)
    va = (1.0 + r902 - t902 + delta) / (2.0 * r90)

    beta_r90 = torch.clamp(beta - r90, min=0.0005)
    vb = torch.sqrt(beta * (va - r90) / (va * beta_r90))

    vbNN = vb ** (N.unsqueeze(1) - 1.0)
    vbNNinv = 1.0 / vbNN
    vainv = 1.0 / va

    vb_diff = vbNN - vbNNinv
    s1 = ta * t90 * vb_diff
    s2 = ta * (va - vainv)
    s3 = va * vbNN - vainv * vbNNinv - r90 * vb_diff

    rho = ra + s1 / s3
    if print_both:
      tau = s2 / s3

      return rho, tau
    else:
      return rho



def prospect5b(traits, N, CoefMat=None, alpha=40.0, print_both=True, with_power = False):
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


    device = traits.device
    n = to_device(CoefMat[:, 1], device) ## first is WL, second is n
    alpha = to_device(alpha, device)
    N = to_device(N, device) 
    th_CoefMat = to_device(CoefMat[:, 2:], device) ## rest are absorbtion coefs

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
    if device == 'cpu':
      if with_power == False:
        trans_rtm = (1 - k_pd) * torch.exp(-k_pd) + k_pd**2 * exp1(k_pd) #exp1_pytorch
      else:
        trans_rtm = (1 - k_pd) * torch.exp(-k_pd) + k_pd**2 * exp1_pytorch(k_pd) #exp1_pytorch

    else:
      trans_rtm = (1 - k_pd) * torch.exp(-k_pd) + k_pd**2 * exp1_pytorch(k_pd) #exp1_pytorch

  #  import pdb; pdb.set_trace() ## debugger

    return plate_torch(trans_rtm, r12, t12, t21, r21, xp, yp, N, print_both)




def prospectd(traits, N, CoefMat=None, alpha=40.0, print_both=True,with_power = False):
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
        CoefMat = load_prospectd_matrix()  # tensor of shape [2101, 7] => [n, Cab, Car, Cbrown, Cw, Cm,Canth]
        
    device = traits.device
    n = to_device(CoefMat[:, 1], device) ## first is WL, second is n
    alpha = to_device(alpha, device)
    N = to_device(N, device) 
    th_CoefMat = to_device(CoefMat[:, 2:], device) ## rest are absorbtion coefs


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
    if device == 'cpu':
      if with_power == False:
        trans_rtm = (1 - k_pd) * torch.exp(-k_pd) + k_pd**2 * exp1(k_pd) #exp1_pytorch
      else:
        trans_rtm = (1 - k_pd) * torch.exp(-k_pd) + k_pd**2 * exp1_pytorch(k_pd) #exp1_pytorch

    else:
      trans_rtm = (1 - k_pd) * torch.exp(-k_pd) + k_pd**2 * exp1_pytorch(k_pd) #exp1_pytorch
    return plate_torch(trans_rtm, r12, t12, t21, r21, xp, yp, N, print_both)


def prospectpro(traits, N, CoefMat=None, alpha=40.0, print_both=True, with_power = False):
    """
    PROSPECT-PRO model.

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
        CoefMat = load_prospectpro_matrix()  # tensor of shape [2101, 8] => [n, Cab, Car,Cbrown, Cw, Canth, prot,Cbc]
        
    device = traits.device
    n = to_device(CoefMat[:, 1], device) ## first is WL, second is n
    alpha = to_device(alpha, device)
    N = to_device(N, device) 
    th_CoefMat = to_device(CoefMat[:, 2:], device) ## rest are absorbtion coefs


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
    if device == 'cpu':
      if with_power == False:
        trans_rtm = (1 - k_pd) * torch.exp(-k_pd) + k_pd**2 * exp1(k_pd) #exp1_pytorch
      else:
        trans_rtm = (1 - k_pd) * torch.exp(-k_pd) + k_pd**2 * exp1_pytorch(k_pd) #exp1_pytorch

    else:
      trans_rtm = (1 - k_pd) * torch.exp(-k_pd) + k_pd**2 * exp1_pytorch(k_pd) #exp1_pytorch
    return plate_torch(trans_rtm, r12, t12, t21, r21, xp, yp, N, print_both)