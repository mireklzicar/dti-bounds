# Performance Bounds for Joint Embedding Models in Drug-Target Interaction Prediction

This repository contains the code and data for analyzing fundamental performance limits of contrastive drug-target interaction (DTI) models. The work demonstrates that binding affinities in standard benchmarks systematically violate basic metric axioms, establishing theoretical upper bounds on achievable performance.

## Overview

Joint embedding models like ConPlex, Drug CLIP, and SPRINT embed ligands and proteins in shared vector spaces, scoring interactions via geometric proximity. However, our analysis reveals that benchmark datasets (BindingDB, KIBA, DAVIS) contain systematic violations of metric axioms such as the triangle inequality. This creates fundamental performance ceilings that no contrastive model can exceed, regardless of architectural sophistication.

### Key Findings

- **Metric violations**: Binding affinities in benchmark datasets violate basic metric axioms
- **Performance bounds**: Even perfect optimization faces dataset-dependent performance ceilings
- **Dimensionality effects**: Higher dimensions relax these constraints significantly
- **Three behavioral regimes**: Metric-friendly, dimensionality-limited, and geometry-limited datasets

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/dti-bounds.git
cd dti-bounds

# Install dependencies
pip install -r requirements.txt

# Optional: Install in development mode
pip install -e .
```

### Requirements

- Python 3.8+
- PyTorch (GPU recommended for large datasets)
- See `requirements.txt` for complete dependencies

## Repository Structure

After running the reorganization script:

```
dti-bounds/
├── src/
│   ├── data/                    # Data downloading and preprocessing
│   │   └── download_datasets.py
│   ├── analysis/               # Metric violation analysis
│   │   └── triangle_inequality.py
│   ├── experiments/            # MDS implementations
│   │   ├── dot_product.py      # Dot-product similarity MDS
│   │   └── euclidean_mds.py    # Euclidean distance MDS (SMACOF)
│   └── utils/                  # Utility functions
├── scripts/                    # Experiment runners
│   ├── run_dot_experiments.py
│   └── run_euclidean_experiments.py
├── data/                       # Data storage
│   ├── datasets/              # Downloaded datasets
│   └── violations/            # Triangle inequality violations
└── results/                   # Experimental results
```

## Usage

### 1. Reorganize Repository Structure

First, reorganize the repository to the clean structure:

```bash
python reorganize_repo.py
```

### 2. Download Datasets

Download and preprocess the benchmark datasets:

```bash
python src/data/download_datasets.py
```

This downloads and processes:
- BindingDB (IC50, Kd, Ki)
- DAVIS
- KIBA

### 3. Analyze Metric Violations

Find triangle inequality violations in the datasets:

```bash
python src/analysis/triangle_inequality.py data/datasets/BindingDB_IC50/sample.csv --output data/violations/BindingDB_IC50.csv
```

### 4. Compute Performance Bounds

#### Dot Product Models (ConPlex, Drug CLIP style)

For single experiment:
```bash
python src/experiments/dot_product.py \
    -i data/datasets/DAVIS/sample.csv \
    -o results/dot/DAVIS_dim64 \
    --dim 64 \
    -e 3000 \
    --lr 0.1
```

For comprehensive experiments across all datasets and dimensions:
```bash
python scripts/run_dot_experiments.py
```

#### Euclidean Distance Models (SMACOF)

For single experiment:
```bash
python src/experiments/euclidean_mds.py \
    -i data/datasets/DAVIS/sample.csv \
    -o results/euclidean/DAVIS_dim64 \
    --dim 64 \
    -m 300
```

For comprehensive experiments:
```bash
python scripts/run_euclidean_experiments.py
```

## Dataset Description

### BindingDB
- **IC50**: Half-maximal inhibitory concentration measurements
- **Kd**: Dissociation constant measurements  
- **Ki**: Inhibition constant measurements

### KIBA
- Kinase inhibitor bioactivity scores (fused from multiple assays)
- 118,254 drug-target pairs across 2,111 ligands and 229 kinases

### DAVIS
- Complete kinase-inhibitor interaction matrix
- 30,056 pairs across 68 kinases and 442 inhibitors

## Understanding Results

### Performance Bound Categories

1. **Metric-friendly datasets** (BindingDB IC50, Ki)
   - Near-perfect reconstruction at low dimensions (d=2-4)
   - High measurement consistency

2. **Dimensionality-limited datasets** (BindingDB Kd, KIBA)
   - Require substantial capacity (d≥64) but achieve strong performance
   - Complex but learnable interaction patterns

3. **Geometry-limited datasets** (DAVIS)
   - Fundamental performance ceilings even at high dimensions
   - Inherent non-metric structure in biochemical data

### Output Files

Each experiment produces:
- `embeddings.pt`: Full embedding tensor
- `drug_embeddings.pt`: Drug embeddings only
- `target_embeddings.pt`: Target embeddings only
- `*_mapping.csv`: ID to index mappings
- `metrics.json`: Performance metrics and parameters

## Key Scripts

### [`src/experiments/dot_product.py`](src/experiments/dot_product.py)
GPU-accelerated dot-product similarity optimization with SGD. Minimizes squared reconstruction error for inner-product models.

### [`src/experiments/euclidean_mds.py`](src/experiments/euclidean_mds.py)
SMACOF (Stress Majorization) implementation for Euclidean distance embeddings. Uses chunked GPU operations for memory efficiency.

### [`src/analysis/triangle_inequality.py`](src/analysis/triangle_inequality.py)
Identifies 2×2 bicliques where rectangular inequality violations occur, proving non-embeddability in metric spaces.

## Performance Bounds Summary

| Dataset | Method | d=2 | d=16 | d=64 | d=128 |
|---------|--------|-----|------|------|-------|
| DAVIS | Dot Product ρ | 0.51 | 0.68 | 0.76 | 0.78 |
| DAVIS | Euclidean ρ | - | 0.64 | 0.73 | 0.76 |
| KIBA | Dot Product ρ | 0.70 | 0.92 | 0.98 | 0.98 |
| BindingDB Kd | Dot Product ρ | 0.48 | 0.78 | 0.89 | 0.94 |

*ρ = Spearman correlation (rank correlation)*

## Citation

If you use this code or data in your research, please cite:

```bibtex
@article{lzicar2025bounds,
  title={Performance Bounds for Joint Embedding Models in Drug-Target Interaction Prediction},
  author={Lžičař, Miroslav},
  journal={Preprint},
  year={2025}
}
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or issues, please open a GitHub issue or contact:
- Miroslav Lžičař: miroslav.lzicar@deepmedchem.com
- Deep MedChem: https://deepmedchem.com/