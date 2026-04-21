# PSTIF-WRO: Partitioned Spatio-Temporal Intuitionistic Fuzzy Wasserstein Robust Optimization

A Wasserstein Distance-based Spatio-Temporal Intuitionistic Fuzzy Robust Ranking Framework

## Project Overview

PSTIF-WRO is an innovative deep learning framework that integrates:
- **Intuitionistic Fuzzy Sets (IFS)**: Handling uncertainty and fuzziness
- **Optimal Transport Theory (OT)**: Wasserstein distance for measure comparison
- **Graph Neural Networks (GNN)**: Partitioned Wasserstein graph convolution
- **Robust Optimization**: Wasserstein Critic network ensuring adversarial robustness

## Core Innovations

1. **IF-Measure Embedding Layer**: Maps raw features to probability measures on the simplex
2. **PW-GCN**: Measure-valued message passing on partitioned graphs
3. **Wasserstein Critic**: 1-Lipschitz constraint via gradient penalty
4. **Contrastive Completion Mechanism**: Missing value imputation based on InfoNCE

## Datasets

The project supports 6 cross-domain datasets:

| Dataset | Scale | Characteristics |
|---------|-------|-----------------|
| Heart Disease | 303 | High missing rate, medical diagnosis |
| Sepsis Survival | 110K | Spatio-temporal characteristics, survival prediction |
| German Credit | 1K | Small sample, credit assessment |
| Credit Card Fraud | 284K | Extreme imbalance, fraud detection |
| Amazon Electronics | 1.9M | Large-scale recommendation |
| NYC Taxi | 3M | Spatio-temporal data, trip prediction |


## Project Structure
```
pstif_wro/
├── src/
│   ├── models/           # Model definitions
│   │   ├── pstif_wro.py           # Main model
│   │   ├── if_measure_embedding.py # IF measure embedding
│   │   ├── pw_gcn.py              # Partitioned Wasserstein GCN
│   │   ├── wasserstein_critic.py  # Wasserstein Critic
│   │   └── contrastive_completion.py # Contrastive completion
│   ├── layers/           # Custom layers
│   │   ├── sinkhorn.py           # Sinkhorn distance
│   │   ├── gradient_penalty.py   # Gradient penalty
│   │   └── wasserstein_pooling.py # Wasserstein pooling
│   ├── data/             # Data loading
│   │   ├── heart_disease.py
│   │   ├── sepsis.py
│   │   ├── german_credit.py
│   │   ├── credit_card.py
│   │   ├── amazon_electronics.py
│   │   ├── nyc_taxi.py
│   │   └── data_loader.py
│   ├── training/         # Training modules
│   │   ├── trainer.py
│   │   ├── evaluator.py
│   │   └── losses.py
│   └── utils/            # Utility functions
│       ├── config.py
│       ├── logger.py
│       ├── metrics.py
│       ├── seed.py
│       └── visualization.py
├── configs/              # Configuration files
├── scripts/              # Scripts
├── train.py             # Training script
├── evaluate.py          # Evaluation script
├── requirements.txt     # Dependencies
└── README.md           # Documentation
```

## Installation

```bash
# Clone repository
git clone <repository_url>
cd pstif_wro

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# Or venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Training

```bash
# Train on Heart Disease dataset
python train.py --dataset heart_disease --data_path ./data/raw --epochs 200

# Train on German Credit dataset
python train.py --dataset german_credit --data_path ./data/raw --epochs 300

# Use custom configuration
python train.py --config configs/custom.yaml
```

### Evaluation

```bash
# Evaluate model
python evaluate.py \
    --checkpoint outputs/checkpoints/best_model.pth \
    --dataset heart_disease \
    --data_path ./data/raw

# Adversarial robustness testing
python evaluate.py \
    --checkpoint outputs/checkpoints/best_model.pth \
    --dataset credit_card \
    --data_path ./data/raw \
    --adversarial_test
```

## Configuration

Model configurations are managed via YAML files. Key parameters include:

### Model Parameters
- `hidden_dim`: Hidden layer dimension
- `measure_dim`: Measure dimension (default 3: μ, ν, π)
- `dropout`: Dropout probability
- `use_contrastive`: Whether to use contrastive completion
- `use_robust`: Whether to use robust layers

### Training Parameters
- `epochs`: Training epochs
- `batch_size`: Batch size
- `learning_rate`: Learning rate
- `optimizer`: Optimizer type
- `scheduler`: Learning rate scheduler

### Loss Weights
- `lambda_completion`: Completion loss weight
- `lambda_robust`: Robust loss weight
- `lambda_gp`: Gradient penalty weight

## Theoretical Contributions

1. **Differential Geometry of IF-Measure Spaces**: First rigorous embedding of intuitionistic fuzzy numbers into Wasserstein space
2. **OT Equivalence of Differentiable Robust Optimization**: Proving that traditional robust optimization is equivalent to measure projection on Wasserstein balls
3. **Message Passing Upper Bound on Partitioned Graphs**: Proving that PW-GCN has expressive power no lower than the Weisfeiler-Lehman test

## Citation

If this project is helpful to your research, please cite:

```bibtex
@article{pstif_wro_2024,
  title={PSTIF-WRO: Partitioned Spatio-Temporal Intuitionistic Fuzzy Wasserstein Robust Optimization},
  author={Your Name},
  journal={Frontiers of Computer Science},
  year={2024}
}
```

## License

MIT License
```

**Key translation notes:**
- Preserved all code blocks, directory trees, and BibTeX entries exactly as they were
- Translated technical terms using standard academic English conventions (e.g., "鲁棒优化" → "Robust Optimization", "梯度惩罚" → "gradient penalty", "消息传递" → "message passing")
- Kept dataset names and file paths unchanged since they are proper nouns/code references
- Maintained the original Markdown formatting, table alignment, and comment styles in shell commands
