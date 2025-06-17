"""
Evaluation metrics for DTI embedding models.
"""
import numpy as np
from scipy.stats import spearmanr, pearsonr

def evaluate_embeddings(predictions, ground_truth):
    """
    Evaluate embedding quality by correlation metrics.
    
    Args:
        predictions: Predicted affinity/similarity values
        ground_truth: True affinity/similarity values from dataset
        
    Returns:
        dict: Dictionary containing Pearson and Spearman correlations
    """
    # Calculate correlations
    pearson_corr, p_value_p = pearsonr(predictions, ground_truth)
    spearman_corr, p_value_s = spearmanr(predictions, ground_truth)
    
    metrics = {
        'pearson_r': float(pearson_corr),
        'pearson_p_value': float(p_value_p),
        'spearman_rho': float(spearman_corr),
        'spearman_p_value': float(p_value_s),
        'mse': float(np.mean((predictions - ground_truth) ** 2)),
        'mae': float(np.mean(np.abs(predictions - ground_truth)))
    }
    
    return metrics


def calculate_dot_product_similarity(drug_embeddings, target_embeddings, drug_idx, target_idx):
    """
    Calculate dot product similarity between drugs and targets.
    
    Args:
        drug_embeddings: Tensor of drug embeddings (n_drugs, dim)
        target_embeddings: Tensor of target embeddings (n_targets, dim)
        drug_idx: Drug indices for evaluation pairs
        target_idx: Target indices for evaluation pairs
        
    Returns:
        numpy.ndarray: Dot product similarity scores
    """
    # Extract embeddings for evaluation pairs
    drug_vecs = drug_embeddings[drug_idx]
    target_vecs = target_embeddings[target_idx]
    
    # Calculate dot product
    similarities = np.sum(drug_vecs * target_vecs, axis=1)
    
    return similarities


def calculate_euclidean_distance(drug_embeddings, target_embeddings, drug_idx, target_idx):
    """
    Calculate Euclidean distances between drugs and targets.
    
    Args:
        drug_embeddings: Tensor of drug embeddings (n_drugs, dim)
        target_embeddings: Tensor of target embeddings (n_targets, dim)
        drug_idx: Drug indices for evaluation pairs
        target_idx: Target indices for evaluation pairs
        
    Returns:
        numpy.ndarray: Euclidean distance scores
    """
    # Extract embeddings for evaluation pairs
    drug_vecs = drug_embeddings[drug_idx]
    target_vecs = target_embeddings[target_idx]
    
    # Calculate Euclidean distances
    distances = np.sqrt(np.sum((drug_vecs - target_vecs) ** 2, axis=1))
    
    return distances


def print_metrics(metrics):
    """
    Pretty-print evaluation metrics.
    
    Args:
        metrics: Dictionary of metric values
    """
    print("\nEvaluation Metrics:")
    print(f"  Pearson r:     {metrics['pearson_r']:.4f} (p={metrics['pearson_p_value']:.4e})")
    print(f"  Spearman rho:  {metrics['spearman_rho']:.4f} (p={metrics['spearman_p_value']:.4e})")
    print(f"  MSE:           {metrics['mse']:.4f}")
    print(f"  MAE:           {metrics['mae']:.4f}")