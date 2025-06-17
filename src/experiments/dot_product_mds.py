#!/usr/bin/env python3
import argparse
import torch
import pandas as pd
import numpy as np
import os
import json
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='Learn embeddings for drugs and targets using dot product similarity.'
    )
    parser.add_argument(
        '-i', '--input', required=True,
        help='CSV file with columns Drug_ID, Target_ID, Y'
    )
    parser.add_argument(
        '-d', '--dim', type=int, default=2,
        help='Embedding dimension'
    )
    parser.add_argument(
        '-e', '--epochs', type=int, default=10000,
        help='Number of training iterations'
    )
    parser.add_argument(
        '--lr', type=float, default=0.1,
        help='Learning rate'
    )
    parser.add_argument(
        '--chunk_size', type=int, default=100000,
        help='Number of pairs per chunk for loss computation'
    )
    parser.add_argument(
        '-o', '--output_dir', required=True,
        help='Directory to save all output files'
    )
    return parser.parse_args()



def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    df = pd.read_csv(args.input)
    y_vals = df['Y_similarity'].astype(np.float32)
    df['Y_processed'] = y_vals

    # Index entities
    drugs = df['Drug_ID'].unique()
    targets = df['Target_ID'].unique()
    i_drug = {d:i for i,d in enumerate(drugs)}
    i_targ = {t:i for i,t in enumerate(targets)}
    n_drugs, n_targets = len(drugs), len(targets)
    n_total = n_drugs + n_targets

    # Build pair index tensors
    i_idx = torch.tensor([i_drug[d] for d in df['Drug_ID']], dtype=torch.long, device=device)
    j_idx = torch.tensor([n_drugs + i_targ[t] for t in df['Target_ID']], dtype=torch.long, device=device)
    delta = torch.tensor(df['Y_processed'].values, dtype=torch.float32, device=device)

    # Initialize embeddings
    X = torch.randn(n_total, args.dim, device=device, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([X], lr=args.lr)

    # Training loop with early stopping
    best_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    patience_limit = 1000
    min_improvement = 10.0  # Minimum improvement threshold
    
    for epoch in range(1, args.epochs+1):
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)

        # Chunked loss accumulation
        for start in range(0, len(i_idx), args.chunk_size):
            end = start + args.chunk_size
            bi, bj = i_idx[start:end], j_idx[start:end]
            bd = delta[start:end]
            
            xi, xj = X[bi], X[bj]
            # Only use dot product similarity
            err = ((xi * xj).sum(dim=1) - bd) ** 2
            total_loss += err.sum() * 0.5

        # Backprop
        total_loss.backward()
        optimizer.step()
        
        current_loss = total_loss.item()
        
        # Track best loss and epoch
        if current_loss < best_loss:
            best_loss = current_loss
            best_epoch = epoch
        
        # Early stopping logic
        if best_loss - current_loss > min_improvement:
            best_loss = current_loss
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience_limit:
            print(f"Early stopping at epoch {epoch} - no improvement > {min_improvement} for {patience_limit} epochs")
            break
            
        if epoch % 50 == 0 or epoch in (1, args.epochs):
            print(f"Epoch {epoch:4d}/{args.epochs}  Loss: {current_loss:.6f} (best: {best_loss:.6f} at epoch {best_epoch}) (patience: {patience_counter}/{patience_limit})")

    # Total epochs actually run
    total_epochs = epoch
    
    # Save embeddings
    X_final = X.detach().cpu()
    
    # Drugs
    drug_emb_tensor = X_final[:n_drugs]
    drug_pt = os.path.join(args.output_dir, "drug_embeddings.pt")
    torch.save(drug_emb_tensor, drug_pt)
    print(f"Saved drug embeddings to {drug_pt}")
    
    # Save drug ID mapping
    drug_mapping = pd.DataFrame(list(i_drug.items()), columns=["Drug_ID", "Index"])
    drug_mapping_path = os.path.join(args.output_dir, "drug_mapping.csv")
    drug_mapping.to_csv(drug_mapping_path, index=False)
    print(f"Saved drug ID mapping to {drug_mapping_path}")
    
    # Targets
    targ_emb_tensor = X_final[n_drugs:]
    targ_pt = os.path.join(args.output_dir, "target_embeddings.pt")
    torch.save(targ_emb_tensor, targ_pt)
    print(f"Saved target embeddings to {targ_pt}")
    
    # Save target ID mapping
    target_mapping = pd.DataFrame(list(i_targ.items()), columns=["Target_ID", "Index"])
    target_mapping_path = os.path.join(args.output_dir, "target_mapping.csv")
    target_mapping.to_csv(target_mapping_path, index=False)
    print(f"Saved target ID mapping to {target_mapping_path}")
    
    # Full tensor
    full_pt = os.path.join(args.output_dir, "embeddings.pt")
    torch.save(X_final, full_pt)
    print(f"Saved full embeddings to {full_pt}")
    
    # Save metrics and parameters to JSON
    metrics = {
        'best_loss': best_loss,
        'best_epoch': best_epoch,
        'total_epochs': total_epochs,
        'parameters': {
            'dim': args.dim,
            'epochs': args.epochs,
            'lr': args.lr,
            'chunk_size': args.chunk_size,
            'input_file': args.input,
            'n_drugs': n_drugs,
            'n_targets': n_targets,
            'patience_limit': patience_limit,
            'min_improvement': min_improvement,
            'early_stopped': total_epochs < args.epochs
        }
    }
    
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

if __name__ == '__main__':
    main()