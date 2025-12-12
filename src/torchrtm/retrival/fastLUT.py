import torch
import torch.nn as nn
import torch.utils.benchmark as benchmark
from torch.nn import functional as F
from torchrtm.utils.torch_utils import normalize_parameters

import torch
from tqdm import tqdm

@torch.no_grad()
def Torchlut_pred(
    xb, xq, y, 
    k=5, 
    distance_order=2, 
    xb_block=1000, 
    batch_size=200, 
    device="cuda"
):
    """
    Perform fast lookup-table (LUT)-style prediction using PyTorch with block-wise distance computation.

    Parameters
    ----------
    xb : torch.Tensor
        Database feature matrix of shape (N, D).
    xq : torch.Tensor
        Query feature matrix of shape (M, D).
    y : torch.Tensor
        Target values corresponding to xb, shape (N,) or (N, D_out).
    k : int, default=5
        Number of nearest neighbors to use.
    distance_order : int, default=2
        Distance metric order (e.g. 2 for Euclidean, 1 for Manhattan).
    xb_block : int, default=100
        Block size for processing xb to manage memory.
    batch_size : int, default=200
        Batch size for query processing.
    device : str, default="cuda"
        Device for computation.

    Returns
    -------
    preds : torch.Tensor
        Predicted values for each query, shape (M, D_out) or (M,) if D_out == 1.
    """
    xb, xq, y = xb.to(device), xq.to(device), y.to(device)
    N, D = xb.shape
    M = xq.shape[0]
    
    # Ensure y has the correct dimension
    if y.ndim == 1:
        y = y.unsqueeze(-1)
    D_out = y.shape[1]
    
    preds = torch.empty(M, D_out, device=device)
    
    # Process query points in batches
    for qs in tqdm(range(0, M, batch_size), desc="Processing query blocks"):
        qe = min(qs + batch_size, M)
        xq_chunk = xq[qs:qe]
        qc = xq_chunk.shape[0]
        
        # Store all distance blocks and corresponding indices
        all_dists = []
        all_indices = []
        
        if distance_order == 2:
            # Precompute squared norms for queries
            xq_norm = (xq_chunk ** 2).sum(dim=1, keepdim=True)
        
        # Loop over database blocks
        for start in range(0, N, xb_block):
            end = min(start + xb_block, N)
            xb_chunk = xb[start:end]
            
            if distance_order == 2:
                # Efficient Euclidean distance computation
                xb_norm = (xb_chunk ** 2).sum(dim=1, keepdim=True).T
                dist = xq_norm + xb_norm - 2 * (xq_chunk @ xb_chunk.T)
                dist = torch.clamp(dist, min=0).sqrt()
            else:
                # General distance metric
                dist = torch.cdist(xq_chunk, xb_chunk, p=distance_order)
            
            # Adjust indices to global positions
            indices = torch.arange(start, end, device=device).unsqueeze(0).expand(qc, -1)
            
            all_dists.append(dist)
            all_indices.append(indices)
        
        # Concatenate all blocks along the database dimension
        all_dists = torch.cat(all_dists, dim=1)    # [qc, N]
        all_indices = torch.cat(all_indices, dim=1)  # [qc, N]
        
        # Select global top-k nearest neighbors
        topk_dist, order = torch.topk(all_dists, k, dim=1, largest=False)
        topk_idx = torch.gather(all_indices, 1, order)
        
        # Aggregate predictions from nearest neighbors
        preds[qs:qe] = y[topk_idx].mean(dim=1)
    
    # Squeeze the last dimension if output is scalar
    if D_out == 1:
        preds = preds.squeeze(-1)
    
    return preds
