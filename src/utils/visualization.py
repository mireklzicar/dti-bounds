"""
Visualization utilities for DTI embedding analysis.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import torch

def plot_correlation(true_values, predicted_values, title, output_path=None):
    """
    Create a scatter plot of true vs. predicted values with regression line.
    
    Args:
        true_values: Ground truth values
        predicted_values: Model predictions
        title: Plot title
        output_path: Path to save the figure
    """
    plt.figure(figsize=(8, 6))
    
    # Create the scatter plot
    sns.regplot(x=true_values, y=predicted_values, scatter_kws={'alpha': 0.5})
    
    # Add perfect correlation line
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    lim_min = min(x_min, y_min)
    lim_max = max(x_max, y_max)
    plt.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', alpha=0.5)
    
    # Beautify
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    plt.grid(alpha=0.3)
    
    # Save or show
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_embeddings(drug_embeddings, target_embeddings, output_path=None, 
                        method='tsne', perplexity=30, n_iter=1000):
    """
    Visualize drug and target embeddings in 2D.
    
    Args:
        drug_embeddings: Drug embedding matrix
        target_embeddings: Target embedding matrix
        output_path: Path to save the figure
        method: Dimensionality reduction method ('tsne' or 'pca')
        perplexity: t-SNE perplexity parameter
        n_iter: t-SNE number of iterations
    """
    # Convert to numpy if needed
    if isinstance(drug_embeddings, torch.Tensor):
        drug_embeddings = drug_embeddings.detach().cpu().numpy()
    if isinstance(target_embeddings, torch.Tensor):
        target_embeddings = target_embeddings.detach().cpu().numpy()
    
    # Combine embeddings
    all_embeddings = np.vstack([drug_embeddings, target_embeddings])
    
    # Apply dimensionality reduction if needed
    if all_embeddings.shape[1] > 2:
        if method.lower() == 'tsne':
            reducer = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
            reduced_embeddings = reducer.fit_transform(all_embeddings)
        else:
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2)
            reduced_embeddings = reducer.fit_transform(all_embeddings)
    else:
        reduced_embeddings = all_embeddings
    
    # Split back into drug and target embeddings
    n_drugs = drug_embeddings.shape[0]
    drug_reduced = reduced_embeddings[:n_drugs]
    target_reduced = reduced_embeddings[n_drugs:]
    
    # Create plot
    plt.figure(figsize=(10, 8))
    plt.scatter(drug_reduced[:, 0], drug_reduced[:, 1], 
                c='blue', alpha=0.6, label='Drugs', s=30)
    plt.scatter(target_reduced[:, 0], target_reduced[:, 1], 
                c='red', alpha=0.6, label='Targets', s=50)
    
    plt.title(f'Drug-Target Embedding Visualization ({method})', fontsize=14)
    plt.xlabel('Dimension 1', fontsize=12)
    plt.ylabel('Dimension 2', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    
    # Save or show
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_performance_by_dimension(dimensions, pearson_scores, spearman_scores, 
                                method_name, dataset_name, output_path=None):
    """
    Plot performance metrics vs. embedding dimension.
    
    Args:
        dimensions: List of embedding dimensions
        pearson_scores: List of Pearson correlation scores
        spearman_scores: List of Spearman correlation scores
        method_name: Name of embedding method
        dataset_name: Name of dataset
        output_path: Path to save the figure
    """
    plt.figure(figsize=(10, 6))
    
    # Plot lines
    plt.plot(dimensions, pearson_scores, 'o-', color='#1f77b4', label='Pearson r')
    plt.plot(dimensions, spearman_scores, 's-', color='#ff7f0e', label='Spearman ρ')
    
    # Beautify
    plt.xscale('log', base=2)
    plt.grid(alpha=0.3)
    plt.xlabel('Embedding Dimension', fontsize=12)
    plt.ylabel('Correlation', fontsize=12)
    plt.title(f'{method_name} Performance on {dataset_name}', fontsize=14)
    plt.legend(fontsize=12)
    
    # Mark key dimensions
    for d in dimensions:
        if d in [2, 16, 64, 128]:
            plt.axvline(x=d, color='gray', linestyle='--', alpha=0.5)
    
    # Set nice axis limits
    plt.ylim(0, 1.05)
    plt.xticks(dimensions)
    
    # Save or show
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_stress_convergence(stress_values, title="SMACOF Stress Convergence", 
                          output_path=None):
    """
    Plot stress convergence curve for SMACOF algorithm.
    
    Args:
        stress_values: List of stress values at each iteration
        title: Plot title
        output_path: Path to save the figure
    """
    plt.figure(figsize=(10, 6))
    
    iterations = np.arange(1, len(stress_values) + 1)
    plt.plot(iterations, stress_values, 'o-')
    
    plt.xlabel('Iteration')
    plt.ylabel('Stress')
    plt.title(title)
    plt.grid(alpha=0.3)
    
    # Log scale often helps visualize convergence better
    plt.yscale('log')
    
    # Save or show
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()