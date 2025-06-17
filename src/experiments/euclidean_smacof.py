#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from tqdm import trange, tqdm
import json  # Added for JSON metrics
import math 

def parse_args():
    p = argparse.ArgumentParser(
        description="Bipartite SMACOF (drug–target) on GPU with chunked Guttman updates"
    )
    p.add_argument('-i','--input',    required=True, help='CSV with Drug_ID,Target_ID,Y')
    p.add_argument('-d','--dim',      type=int, default=2, help='embedding dimension')
    p.add_argument('-m','--max_iter', type=int, default=100, help='max SMACOF iterations')
    p.add_argument('-e','--eps',      type=float, default=1e-4, help='stress tol (abs drop)')
    p.add_argument('-b','--batch',    type=int, default=1024,
                   help='chunk size for building B (rows/cols)')
    p.add_argument('-o','--output',   required=True, help='output directory')
    return p.parse_args()


def build_full_matrices(df, n_drugs, n_targets, device):
    """Return dissimilarity Δ_full and weight W as (n_total x n_total) tensors."""
    n = n_drugs + n_targets
    Δ = torch.zeros((n,n), dtype=torch.float32, device=device)
    W = torch.zeros_like(Δ)
    
    # Precompute all indices at once
    drug_indices = torch.tensor([drug2idx[d] for d in tqdm(df.Drug_ID.values)], device=device)
    target_indices = torch.tensor([n_drugs + targ2idx[t] for t in tqdm(df.Target_ID.values)], device=device)
    y_values = torch.tensor(df.Y.values, dtype=torch.float32, device=device)
    
    # Set values in one batch operation
    Δ[drug_indices, target_indices] = y_values
    Δ[target_indices, drug_indices] = y_values
    W[drug_indices, target_indices] = 1.0
    W[target_indices, drug_indices] = 1.0
    
    return Δ, W

def save_embeddings(X, n_drugs, output_dir):
    """Save embeddings to files."""
    X_cpu = X.detach().cpu()
    drug_emb_tensor = X_cpu[:n_drugs]
    target_emb_tensor = X_cpu[n_drugs:]
    
    # Save tensors
    torch.save(drug_emb_tensor, os.path.join(output_dir, 'drug_embeddings.pt'))
    torch.save(target_emb_tensor, os.path.join(output_dir, 'target_embeddings.pt'))
    torch.save(X_cpu, os.path.join(output_dir, 'embeddings.pt'))

def smacof_gpu(Δ: Tensor, W: Tensor, dim: int, max_iter: int, eps: float, batch: int, n_drugs: int, output_dir: str, verbose: bool=True):
    '''SMACOF via Guttman transform, all on GPU, chunked B‑matrix build'''
    device = Δ.device
    n = Δ.size(0)

    # build V = diag(W.sum(1)) - W
    deg = W.sum(1)
    λ   = 1e-6 * deg.mean()           # scale reg to data
    V   = torch.diag(deg) - W
    V  += λ * torch.eye(n, device=device, dtype=V.dtype)
    Vp  = torch.inverse(V)

    # init X randomly
    X = torch.randn((n, dim), device=device)
    
    # Track best stress and X
    best_stress = float('inf')
    best_X = X.clone()
    best_iter = 0

    # helper to chunked compute B = -w * δ/ d, diag=B.sum(row)
    def make_B(dist):
        B = torch.zeros_like(dist, device=device)
        # process in row‑chunks
        for i0 in range(0, n, batch):
            i1 = min(n, i0+batch)
            # each row‑chunk against ALL columns
            Wij = W[i0:i1]              # (bi,n)
            Δij = Δ[i0:i1]              # (bi,n)
            dij = dist[i0:i1]           # (bi,n)
            mask = Wij>0
            # compute ratio only where W>0
            r   = torch.zeros_like(dij)
            r[mask] = Δij[mask] / dij[mask]
            Bij = -r
            Bij[torch.arange(i0,i1,device=device)-i0, torch.arange(i0,i1,device=device)] = r.sum(dim=1)
            B[i0:i1] = Bij
        return B

    old_stress = None
    for it in trange(max_iter, disable=not verbose, desc='SMACOF'):
        dist = torch.cdist(X, X) + 1e-8      # (n,n)
        dist = dist.clamp(min=1e-6)

        # stress = ½ * ∑_ij w_ij (d_ij(X) - Δ_ij)^2
        diff   = dist - Δ
        stress = 0.5 * (W * diff * diff).sum().item()
        
        # The best stress tracking logic has been moved below

        if old_stress is not None and old_stress - stress < eps and stress < old_stress:
            if verbose:
                print(f"→ converged at iter {it:3d}: stress {stress:.6f}, best stress {best_stress:.6f} at iter {best_iter}")
            break
        if math.isnan(stress):
            if verbose:
                print(f"→ nan at iter {it:3d}: stress {stress:.6f}, best stress {best_stress:.6f} at iter {best_iter}")
            break
        old_stress = stress

        # build weighted B in chunks
        B = make_B(dist)

        # Guttman update
        X = Vp @ (B @ X)

        # Just print progress every 50 iterations
        if it % 50 == 0 and verbose:
            print(f"iter {it:3d}\tstress {stress:.6f}, best stress {best_stress:.6f} at iter {best_iter}")
            
        # Save the best embeddings whenever we find a better stress
        if stress < best_stress:
            best_stress = stress
            best_X = X.clone()
            best_iter = it
            # No saving here - we'll save the best at the end

    return best_X, best_stress, best_iter, it+1

if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    # load
    df = pd.read_csv(args.input)
    df['Y'] = df['Y_distance']
    
    # index drugs & targets
    drugs   = df.Drug_ID.unique()
    targets = df.Target_ID.unique()
    n_drugs, n_targets = len(drugs), len(targets)
    drug2idx = {d:i for i,d in enumerate(drugs)}
    targ2idx = {t:i for i,t in enumerate(targets)}

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device}")

    # build full Δ and W on GPU
    Δ, W = build_full_matrices(df, n_drugs, n_targets, device)

    # run SMACOF
    best_X, best_stress, best_iter, total_iters = smacof_gpu(
        Δ, W,
        dim=args.dim,
        max_iter=args.max_iter,
        eps=args.eps,
        batch=args.batch,
        n_drugs=n_drugs,
        output_dir=args.output,
        verbose=True
    )
    print(f"Done: best stress={best_stress:.6f} at iteration {best_iter}, total iterations={total_iters}")

    # Save only the best embeddings
    save_embeddings(best_X, n_drugs, args.output)
    
    # Save mappings
    pd.DataFrame({'Drug_ID':drugs,   'Index':np.arange(n_drugs)}) \
      .to_csv(os.path.join(args.output,'drug_mapping.csv'),   index=False)
    pd.DataFrame({'Target_ID':targets,'Index':np.arange(n_targets)}) \
      .to_csv(os.path.join(args.output,'target_mapping.csv'), index=False)
    
    # Save final metrics and parameters to JSON
    metrics = {
        'best_stress': best_stress,
        'best_iteration': best_iter,
        'total_iterations': total_iters,
        'parameters': {
            'dim': args.dim,
            'max_iter': args.max_iter,
            'eps': args.eps,
            'batch': args.batch,
            'input_file': args.input,
            'n_drugs': n_drugs,
            'n_targets': n_targets
        }
    }
    
    with open(os.path.join(args.output, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print("All embeddings, mappings, and metrics saved.")