# SEP Forecasting Research - Handoff Documentation

## Overview

This repository contains research on **imbalanced regression for Solar Energetic Particle (SEP) event forecasting**, focusing on predicting rare high-impact space weather events. The work explores novel representation learning techniques, loss functions, and training strategies to handle severe class imbalance (~0.9% rare events).

**Primary Dataset**: SEP-CME (Solar Energetic Particle - Coronal Mass Ejection)  
**Hugging Face**: `erud1t3/dltr_tabular_bench`

---

## Key Research Concepts & File Locations

### 1. Representation Learning: PDS & PDC

#### PDS (Piecewise Distribution Sampling)
**Primary File**: `modules/training/cme_modeling.py`

| Function | Description |
|----------|-------------|
| `pds_loss_vec()` (~line 2338) | Computes PDS loss using pairwise distances between labels and representations |
| `pds_space_norm()` (~line 59) | Normalizes labels for PDS space: `y' = (Dz_max / (ρ * Dy_max)) * y` |
| `train_pds()` (~line 694) | Training function for PDS models |
| `overtrain_pds()` (~line 358) | Overtraining function for PDS |

**Related Training Scripts**:
- `sources/mlp/pds/stage1/stratinj_sepc.py` - Stage 1 PDS training with stratified injection
- `sources/mlp/pds/stage2/` - Stage 2 PDS training scripts

**Supporting Modules**:
- `modules/training/normlayer.py` - NormalizeLayer for PDS normalization

---

#### PDC (Piecewise Distribution Correlation)
**Primary File**: `modules/training/cme_modeling.py`

| Function | Description |
|----------|-------------|
| `pdc_loss_vec()` (~line 1929) | PDC loss using pairwise distance correlation (excludes diagonal terms) |
| `pdc_loss_vec_geo()` (~line 2007) | Geodesic distance version of PDC loss |
| `pdc_loss_linear_vec()` (~line 2099) | Optimized linear version of PDC loss |
| `pdc_loss_linear_m_vec()` (~line 2174) | PDC loss with specific pair selection |

**Key Insight**: PDC computes correlation between pairwise label distances and pairwise representation distances. Loss formula: `1.0 - pcc` where pcc is the Pearson correlation coefficient of distances.

**Related Training Scripts**:
- `sources/mlp/pdc/stage1/stratinj_sepc.py` - Stage 1 PDC training
- `sources/mlp/pdc/stage2/cheat_ae_quc.py` - Stage 2 PDC with autoencoder
- `sources/mlp/neurips/sep_cme/folds_ours_quc_s2_reg.py` - PDC regularization experiments
- `sources/mlp/neurips/sep_cme/folds_ours_quc_s2_reg_geo.py` - Geodesic PDC experiments

**Notebooks**:
- `notebooks/pdc_test.ipynb`
- `notebooks/pdc_test_norm_fix.ipynb`

---

### 2. Gradient Conflicts Handling

**Status**: Research/analysis only (not integrated into main training pipeline)

**Primary Files**:
- `notebooks/analysis_grad_conflicts.ipynb`
- `notebooks/analysis_grad_conflicts_3bins.ipynb`
- `notebooks/analysis_grad_conflicts_lrdecay.ipynb`
- `notebooks/analysis_grad_conflicts_pdc.ipynb`

**Key Function**: `adjust_grads()` (found in notebooks, ~line 1533 in `analysis_grad_conflicts_pdc.ipynb`)

**Formula**:
```
g̃_i = g_i - β * ||g_i|| * Σ_{j≠i} ((cos(g_i, g_j) / ||g_j||) * g_j)
```

**Beta Strategies**:
- `'idea1'`: β = 1
- `'idea2'`: β = 1/(k-1) or 1/num_conflicts
- `'idea12'`: Adaptive based on average absolute pairwise cosine similarity (aapc)
- `'interpolate'`: Interpolates between idea1 and idea2

**Parameters**:
- `check_conflicts`: If True, only adjusts gradients with negative cosine similarity

---

### 3. Multiple Models/Experts (MoE - Mixture of Experts)

**Primary Directory**: `sources/mlp_moe/`

**Directory Structure**:
```
sources/mlp_moe/
├── combiner/      # Router/combiner models that route inputs to experts
├── assemble/      # Assembled MoE models (combined experts)
├── zero_e/        # Experts for samples near zero
├── plus_e/        # Experts for samples above threshold
└── minus_e/       # Experts for samples below threshold
```

**Key Training Scripts**:
| File | Description |
|------|-------------|
| `combiner/folds.py` | Main router training script |
| `combiner/cheat.py`, `cheat2_*.py`, `cheat3*.py` | Various router configurations |
| `assemble/folds_v2.py` | Assembled MoE training |
| `assemble/cheat_v*.py` | Various assembled MoE configurations |

**Key Implementation Details**:
- Router outputs 3 classes (routing probabilities) - `ROUTER_OUTPUT_DIM = 3`
- Uses focal loss for classification (routing)
- Experts trained separately, then combined
- Supports PDC regularization in some variants (`cheat2_pdcaes1.py`, `cheat2_pdcaes2.py`)

**Configuration** (in `modules/shared/sep_globals.py`):
- `ROUTER_OUTPUT_DIM = 3`
- `LOWER_THRESHOLD`, `UPPER_THRESHOLD` define expert ranges

---

### 4. Instance-Dependent Attention Mechanism

**Primary File**: `sources/attm/modules.py`

| Class/Function | Line | Description |
|----------------|------|-------------|
| `AttentionBlock` | ~302 | Base attention block with skip connections |
| `TanhAttentiveBlock` | ~503 | Attention block with tanh-scaled attention scores |
| `create_attentive_model()` | ~23 | Factory function for attention-based models |

**Key Feature**: Attention weights are **instance-dependent**, computed dynamically at inference (not constant after training).

**Formula**:
```
y = w0 + w1 * a1 * x1 + w2 * a2 * x2 + ...
where a_i = tanh(a * attention_score_i)
```

- `a` is a trainable scaling parameter (~line 566)
- Attention scores stored in `attention_scores` attribute (updated each forward pass)
- Models return `{'output': output, 'attention_scores': attention_scores}`

**Training Scripts**:
- `sources/attm/folds.py` - Attention model training
- `sources/attm/folds_ff.py` - Feed-forward attention variant
- `sources/attm/cheat.py` - Evaluation script

**Notebooks**:
- `notebooks/attention_exps.ipynb`

---

### 5. SAM (Sharpness Aware Minimization)

**Primary File**: `modules/training/sam_keras.py`

| Class/Function | Line | Description |
|----------------|------|-------------|
| `SAM` | ~6 | SAM optimizer wrapper class |
| `SAMModel` | ~162 | Custom Keras Model integrating SAM into training step |
| `sam_train_step()` | ~74 | Custom training step function |

**Key Methods**:
- `first_step()`: Perturbs model parameters by `ρ * gradient_norm`
- `second_step()`: Reverts perturbation and applies update

**Implementation Details**:
- Two forward passes: original weights → perturbed weights
- Perturbation: `w + ρ * grad / ||grad||` (normalized) or `w + ρ * grad` (unnormalized)
- Uses perturbed gradients for final update

**Usage**:
- Set `sam_rho > 0` in model creation to enable SAM
- Default `ρ = 0.05`
- Integrated into `create_attentive_model()` and other model builders

**Configuration**:
- `RHO` parameter in `sep_globals.py` (line 59: `RHO = [0]` - disabled by default)

---

## Repository Structure

```
sep-forecasting-research/
├── modules/
│   ├── training/
│   │   ├── cme_modeling.py      # PDS/PDC loss implementations, ModelBuilder ⭐
│   │   ├── ts_modeling.py       # Time series modeling, dataset builders
│   │   ├── sam_keras.py         # SAM optimizer implementation ⭐
│   │   ├── normlayer.py         # NormalizeLayer for PDS
│   │   └── phase_manager.py     # Training phase tracking
│   ├── evaluate/                # Evaluation metrics and utilities
│   ├── reweighting/             # Importance weighting functions
│   └── shared/
│       ├── sep_globals.py       # Main SEP-CME configuration ⭐
│       ├── globals.py           # Base global configuration
│       └── *_globals.py         # Other dataset configs
├── sources/
│   ├── mlp/                     # Standard MLP models
│   │   ├── pds/                 # PDS training scripts
│   │   ├── pdc/                 # PDC training scripts
│   │   └── neurips/sep_cme/     # NeurIPS experiments
│   ├── mlp_moe/                 # Mixture of Experts ⭐
│   │   ├── combiner/            # Router models
│   │   ├── assemble/            # Assembled MoE
│   │   └── *_e/                 # Expert models
│   └── attm/                    # Attention models ⭐
│       └── modules.py           # Attention implementations
├── notebooks/                   # Analysis notebooks
│   ├── analysis_grad_conflicts*.ipynb  # Gradient conflict analysis ⭐
│   ├── pdc_test*.ipynb          # PDC experiments
│   └── attention_exps.ipynb     # Attention experiments
├── aip_scripts/                 # SLURM scripts for AI-Panther cluster
│   ├── slurm_gpu*.sh            # GPU training scripts
│   └── utils/                   # Utility scripts
├── requirements.txt             # Python dependencies
└── README.md                    # Basic repository info
```

---

## Main Entry Points

### Training Scripts

| Task | Script |
|------|--------|
| Base MLP | `sources/mlp/folds.py` |
| SEP-CME with QUC | `sources/mlp/neurips/sep_cme/folds_ours_quc.py` |
| Stage 2 training | `sources/mlp/neurips/sep_cme/folds_ours_quc_s2.py` |
| PDC regularization | `sources/mlp/neurips/sep_cme/folds_ours_quc_s2_reg.py` |
| Geodesic PDC | `sources/mlp/neurips/sep_cme/folds_ours_quc_s2_reg_geo.py` |
| MoE Router | `sources/mlp_moe/combiner/folds.py` |
| MoE Assembled | `sources/mlp_moe/assemble/folds_v2.py` |
| Attention models | `sources/attm/folds.py` |
| PDS Stage 1 | `sources/mlp/pds/stage1/stratinj_sepc.py` |
| PDC Stage 1 | `sources/mlp/pdc/stage1/stratinj_sepc.py` |

### Evaluation Scripts
- `sources/mlp/load_and_eval.py`
- `sources/mlp_moe/combiner/misc/load_and_eval.py`
- `aip_scripts/utils/load_get_decomps.py` (MSE decomposition analysis)

---

## Configuration

**Global Configuration Files**:
- `modules/shared/sep_globals.py` - Main SEP-CME dataset configuration
- `modules/shared/globals.py` - Base global configuration

**Key Hyperparameters** (from `sep_globals.py`):
```python
MLP_HIDDENS = [512, 32, 256, 32, 128, 32, 64, 32]
EMBED_DIM = 32
BATCH_SIZE = 200
EPOCHS = int(2e5)
START_LR = 5e-4
RHO = [0]  # SAM disabled by default
REWEIGHTS = [(0.85, 0.85, 0.0, 0.0)]  # (alpha_mse, alphaV_mse, alpha_pcc, alphaV_pcc)
```

---

## Requirements

See `requirements.txt` for full list. Key dependencies:
- TensorFlow >= 2.19.0
- TensorFlow Addons >= 0.21.0
- TensorFlow Probability >= 0.24.0
- scikit-learn >= 1.3.0
- NumPy >= 1.24.0
- pandas >= 2.0.0
- wandb >= 0.15.0

---

## Key Results

From documentation in `logs/`:
- wPCC regularization improves PCC from 3.7% to 27.4% (all samples)
- Rare events: PCC improves from 35.1% to 70.3%
- MSE reduction: 97.0% on all samples, 84.4% on rare events

---

## Related Repositories

Adapted implementations for representation learning on imbalanced regression:
- **BalancedMSE**: Balanced MSE loss implementations (GAI, BMC, BNI)
- **ConR**: Contrastive Regularizer for imbalanced regression
- **HCA**: Hierarchical Class Attention for imbalanced regression

---

## Contact

For questions about this codebase, please contact the research team.
