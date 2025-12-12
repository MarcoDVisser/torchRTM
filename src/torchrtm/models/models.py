"""
torchrtm.models
---------------

High-level PROSAIL canopy reflectance wrapper for integrated simulation.
This module combines the PROSPECT leaf models with the 4SAIL canopy model.

Authors:
    - Peng Sun
    - Marco D. Visser
"""

import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

# Import leaf models (for PROSPECT)
from torchrtm.leaf.prospect import prospect5b, prospectd, prospectpro
# Import the canopy model (4SAIL)
from torchrtm.canopy.sail import foursail_main
# Import data loader for soil spectra
from torchrtm.data_loader import load_soil_spectra
# Import utility function for device conversion
from torchrtm.utils import to_device


def prosail(
    traits, N, LIDFa, LIDFb, lai, q,
    tts, tto, psi, tran_alpha, psoil,
    batch_size=0, prospect_type='prospect5', lidtype=1
):
    """
    PROSPECT shell function that combines the PROSPECT leaf model and the 4SAIL canopy model.
    
    This function is designed to be fully compatible with existing code.
    If no coefficient matrix is supplied in the underlying leaf functions,
    they will load the default internal data automatically.
    
    Args:
        traits (torch.Tensor): Leaf biochemical traits, shape (batch, n_traits) or (n_traits,)
        N (torch.Tensor): Leaf structure parameter, shape (batch,) or scalar
        LIDFa, LIDFb (torch.Tensor): Leaf angle distribution parameters.
        lai (torch.Tensor): Leaf area index.
        q (torch.Tensor): Hotspot parameter.
        tts (torch.Tensor): Solar zenith angle (degrees).
        tto (torch.Tensor): Sensor viewing angle (degrees).
        psi (torch.Tensor): Relative azimuth angle (degrees).
        tran_alpha (torch.Tensor): Leaf transmission (incidence) parameter.
        psoil (torch.Tensor): Soil moisture fraction.
        batch_size (int): Batch size for computations.
        use_prospectd (bool): Whether to use PROSPECT-D (if True) or PROSPECT-5B.
        lidtype (int): Leaf inclination distribution function type (1–4).
    
    Returns:
        pd.DataFrame or torch.Tensor: Canopy reflectance components.
            In non-batch mode, returns a pandas DataFrame with columns:
            ['rddt', 'rsdt', 'rdot', 'rsot', 'tsd', 'tdd', 'rdd'].
            In batch mode, returns a torch.Tensor.
    """
    device = traits.device
    # Define leaf angle bins (13 values) as per original code:
    litab = torch.tensor([5, 15, 25, 35, 45, 55, 65, 75, 81, 83, 85, 87, 89], device=device)
    pi_val = torch.tensor(torch.pi, device=device)
    rd = pi_val / 180

    # Load soil reflectance spectra from package data;
    # if not provided externally, these are loaded by default.
    dry_soil, wet_soil = load_soil_spectra(device=device)

    # Choose the PROSPECT version based on the flag.
    if prospect_type == 'prospect5d':
        prospect = prospectd  
    elif prospect_type == 'prospect5b':
        prospect =  prospect5b
    elif prospect_type == 'prospectpro':
        prospect =  prospectpro


    with torch.no_grad():
        if traits.ndim == 1:
            # Single sample: ensure traits and N are batched
            traits = traits.unsqueeze(0)
            N = N.unsqueeze(0)

        # Batch mode: ensure batch processing.
        dataset = TensorDataset(traits, N, LIDFa, LIDFb, lai, q, tts, tto, psi, psoil, tran_alpha)
        loader = DataLoader(dataset, batch_size=min(10000, batch_size))
        batch_size = 10000
        num_samples = traits.shape[0]
        results = []

        for i in range(0, num_samples, batch_size):
            b_traits     = traits[i:i+batch_size]
            b_N          = N[i:i+batch_size]
            b_LIDFa      = LIDFa[i:i+batch_size]
            b_LIDFb      = LIDFb[i:i+batch_size]
            b_lai        = lai[i:i+batch_size]
            b_q          = q[i:i+batch_size]
            b_tts        = tts[i:i+batch_size]
            b_tto        = tto[i:i+batch_size]
            b_psi        = psi[i:i+batch_size]
            b_psoil      = psoil[i:i+batch_size]
            b_tran_alpha = tran_alpha[i:i+batch_size]            
            rho, tau = prospect(b_traits, b_N, alpha=b_tran_alpha)
            out = foursail_main(
                rho, tau, b_psoil, b_LIDFa, b_LIDFb, b_lai, b_q,
                b_tts, b_tto, b_psi,
                rd, pi_val, litab, dry_soil, wet_soil,lidtype=lidtype
            )
            results.append(torch.stack(out, dim=0).transpose(1, 2).detach().cpu())
        return torch.cat(results, dim=2)
