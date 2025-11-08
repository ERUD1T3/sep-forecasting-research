"""
MSE Decomposition Analysis for CISIR vs CISIR w/o wPCC

This script computes the Murphy (1988) MSE decomposition to analyze how including
the weighted Pearson Correlation Coefficient (wPCC) regularizer affects the three
components of prediction error for SEP Electron-Channel (SEP-EC) forecasting.

MSE Decomposition (Murphy 1988, Eq. 9):
    MSE(ŷ,y) = (ȳ - ȳ_hat)² + (sd(ŷ) - sd(y))² + 2·sd(ŷ)·sd(y)·(1 - PCC(ŷ,y))

The three terms represent:
    1. Mean mismatch: (ȳ - ȳ_hat)² - error due to bias in predictions
    2. SD mismatch: (sd(ŷ) - sd(y))² - error due to over/under-dispersed predictions
    3. Correlation deficit: 2·sd(ŷ)·sd(y)·(1 - PCC(ŷ,y)) - error due to poor correlation

Additionally, the script reports (1 - PCC) separately to directly show the correlation
degradation, which is particularly important because wPCC regularization directly
affects PCC. This is relevant in the context of Table IV (AORC), which aggregates
two correlation metrics.

Models Compared:
    - CISIR w/o wPCC: Trained with only MSE loss (λ_pcc = 0)
    - CISIR: Trained with MSE + wPCC regularizer (λ_pcc > 0)

Test Subsets:
    - All: All test samples (n=11,839)
    - Rare: High impact events where δ_p ≥ 0.5 (n≈112, 0.9% of test set)

Seeds:
    Analysis is performed across 5 random seeds (456789, 42, 123, 0, 9999)
    and results are averaged to provide robust estimates.

Outputs:
    1. LaTeX table rows for direct copy-paste into paper (with 1-PCC column)
    2. Detailed CSV: All seeds, subsets, and models
    3. Summary CSV: Averaged results with Δ and %Δ

Usage:
    python load_get_decomps.py

Author: NeurIPS 2025 Paper
Date: November 2025
"""

import numpy as np
import tensorflow as tf
import pandas as pd
import sys
import os
from datetime import datetime

# Add current directory to path for module imports (works from root or subdirectory)
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = script_dir if os.path.exists(os.path.join(script_dir, 'modules')) else os.path.abspath(os.path.join(script_dir, '../../..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from modules.shared.globals import *
from modules.training.ts_modeling import build_dataset, create_mlp


def compute_mse_decomposition(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float, float]:
    """
    Compute the three MSE decomposition terms from Murphy (1988).
    
    MSE decomposition: MSE = (ȳ - ȳ_hat)² + (sd(ŷ) - sd(y))² + 2·sd(ŷ)·sd(y)·(1 - PCC(ŷ,y))
    
    Args:
        y_true: Ground truth values (shape: [n_samples, 1] or [n_samples,])
        y_pred: Predicted values (shape: [n_samples, 1] or [n_samples,])
    
    Returns:
        tuple: (mean_term, sd_term, corr_term, one_minus_pcc)
            - mean_term: (ȳ - ȳ_hat)² - mismatch in means
            - sd_term: (sd(ŷ) - sd(y))² - mismatch in standard deviations
            - corr_term: 2·sd(ŷ)·sd(y)·(1 - PCC(ŷ,y)) - correlation deficit term
            - one_minus_pcc: 1 - PCC(ŷ,y) - correlation degradation
    """
    print(f"    Computing MSE decomposition for {len(y_true)} samples...")
    
    # Convert to numpy arrays and flatten if needed
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # Verify shapes match
    assert y_true.shape == y_pred.shape, f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}"
    
    # Compute means
    mean_y = np.mean(y_true)
    mean_y_pred = np.mean(y_pred)
    print(f"    Mean(y_true)={mean_y:.6f}, Mean(y_pred)={mean_y_pred:.6f}")
    
    # Compute standard deviations (using population std with ddof=0 for MSE decomposition)
    sd_y = np.std(y_true, ddof=0)
    sd_y_pred = np.std(y_pred, ddof=0)
    print(f"    SD(y_true)={sd_y:.6f}, SD(y_pred)={sd_y_pred:.6f}")
    
    # Compute Pearson correlation coefficient
    pcc = np.corrcoef(y_true, y_pred)[0, 1]
    print(f"    PCC(y_true, y_pred)={pcc:.6f}")
    
    # Compute the three decomposition terms
    mean_term = (mean_y - mean_y_pred) ** 2
    sd_term = (sd_y_pred - sd_y) ** 2
    one_minus_pcc = 1 - pcc
    corr_term = 2 * sd_y_pred * sd_y * one_minus_pcc
    
    print(f"    Term 1 (mean mismatch): {mean_term:.6f}")
    print(f"    Term 2 (SD mismatch): {sd_term:.6f}")
    print(f"    Term 3 (corr deficit): {corr_term:.6f}")
    print(f"    1 - PCC (corr degradation): {one_minus_pcc:.6f}")
    
    # Verify the decomposition by comparing with direct MSE calculation
    mse_direct = np.mean((y_true - y_pred) ** 2)
    mse_decomp = mean_term + sd_term + corr_term
    decomp_error = abs(mse_direct - mse_decomp)
    
    print(f"    MSE (direct calculation): {mse_direct:.6f}")
    print(f"    MSE (sum of decomposition): {mse_decomp:.6f}")
    print(f"    Decomposition error: {decomp_error:.10f}")
    
    # Warn if decomposition error is too large
    if decomp_error > 1e-6:
        print(f"    WARNING: Large decomposition error detected!")
    
    return mean_term, sd_term, corr_term, one_minus_pcc


def load_model_and_predict(model_path: str, X_test: np.ndarray) -> np.ndarray:
    """
    Load a trained model and make predictions on test data.
    
    This function recreates the model architecture using the same hyperparameters
    from globals, loads the saved weights, and generates predictions.
    
    Args:
        model_path: Path to the saved model weights (.h5 file)
        X_test: Test features array of shape [n_samples, n_features]
    
    Returns:
        y_pred: Model predictions as numpy array of shape [n_samples, output_dim]
    """
    print(f"  Loading model from: {model_path}")
    
    # Get model architecture parameters from globals
    n_features = X_test.shape[1]
    print(f"  Input features: {n_features}")
    print(f"  Model architecture: {MLP_HIDDENS}")
    print(f"  Output dimension: {len(OUTPUTS_TO_USE)}")
    
    # Create model with same architecture used during training
    model = create_mlp(
        input_dim=n_features,
        hiddens=MLP_HIDDENS,
        embed_dim=EMBED_DIM,
        output_dim=len(OUTPUTS_TO_USE),
        dropout=DROPOUT,
        activation=ACTIVATION,
        norm=NORM,
        skip_repr=SKIP_REPR,
        skipped_layers=SKIPPED_LAYERS,
        pretraining=False,
        sam_rho=RHO[0],
        weight_decay=WEIGHT_DECAY
    )
    
    # Load weights from saved checkpoint
    try:
        model.load_weights(model_path)
        print(f"  ✓ Successfully loaded model weights")
    except Exception as e:
        print(f"  ✗ Error loading model weights: {e}")
        raise
    
    # Generate predictions
    # Note: The model has two outputs: [representation, forecast]
    # We only need the forecast (second output) for MSE decomposition
    print(f"  Generating predictions for {X_test.shape[0]} samples...")
    model_output = model.predict(X_test, verbose=0)
    
    # Extract forecast output (second element) from [representation, forecast]
    if isinstance(model_output, list):
        y_pred = model_output[1]  # forecast_head output
        print(f"  ✓ Extracted forecast from multi-output model")
    else:
        y_pred = model_output
    
    # Convert to numpy array
    y_pred = np.asarray(y_pred)
    print(f"  ✓ Predictions generated: shape={y_pred.shape}")
    
    return y_pred


def compute_table_values(
    model_without_wpcc_path: str, 
    model_with_wpcc_path: str, 
    X_test: np.ndarray, 
    y_test: np.ndarray, 
    rare_threshold: float = None
) -> dict:
    """
    Compute all MSE decomposition values needed for the comparison table.
    
    This function loads two models (with and without wPCC regularization),
    generates predictions, and computes MSE decomposition terms for both
    the full test set and the rare samples subset.
    
    Args:
        model_without_wpcc_path: Path to model trained without wPCC (CISIR w/o wPCC)
        model_with_wpcc_path: Path to model trained with wPCC (CISIR)
        X_test: Test features array [n_samples, n_features]
        y_test: Test targets array [n_samples, output_dim]
        rare_threshold: Threshold for defining rare samples (default: MAE_PLUS_THRESHOLD from globals)
    
    Returns:
        dict: Dictionary with structure:
            {
                'all': {
                    'without_wpcc': {'mean': float, 'sd': float, 'corr': float, 'one_minus_pcc': float},
                    'with_wpcc': {'mean': float, 'sd': float, 'corr': float, 'one_minus_pcc': float}
                },
                'rare': {
                    'without_wpcc': {'mean': float, 'sd': float, 'corr': float, 'one_minus_pcc': float},
                    'with_wpcc': {'mean': float, 'sd': float, 'corr': float, 'one_minus_pcc': float}
                }
            }
    """
    # Use global threshold if none provided
    if rare_threshold is None:
        rare_threshold = MAE_PLUS_THRESHOLD
    
    print("\n" + "="*80)
    print("COMPUTING MSE DECOMPOSITION FOR TABLE")
    print("="*80)
    print(f"Rare event threshold: δ ≥ {rare_threshold}")
    
    # Load models and make predictions
    print("\n[STEP 1/4] Loading model WITHOUT wPCC (CISIR w/o wPCC)...")
    y_pred_without = load_model_and_predict(model_without_wpcc_path, X_test)
    
    print("\n[STEP 2/4] Loading model WITH wPCC (CISIR)...")
    y_pred_with = load_model_and_predict(model_with_wpcc_path, X_test)
    
    # Filter for rare samples (high delta_p values)
    rare_mask = y_test[:, 0] >= rare_threshold
    n_total = len(y_test)
    n_rare = np.sum(rare_mask)
    pct_rare = 100 * n_rare / n_total
    
    print(f"\n[STEP 3/4] Identifying rare samples...")
    print(f"  Total test samples: {n_total}")
    print(f"  Rare samples (δ ≥ {rare_threshold}): {n_rare} ({pct_rare:.1f}%)")
    print(f"  Common samples (δ < {rare_threshold}): {n_total - n_rare} ({100 - pct_rare:.1f}%)")
    
    # Initialize results dictionary
    results = {
        'all': {},
        'rare': {}
    }
    
    # Compute decomposition for "All" subset (entire test set)
    print("\n[STEP 4/4] Computing MSE decompositions...")
    print("\n" + "-"*80)
    print(f"MSE DECOMPOSITION - ALL TEST SAMPLES (n={n_total})")
    print("-"*80)
    
    print("\n  Model: CISIR w/o wPCC")
    mean_wo_all, sd_wo_all, corr_wo_all, one_minus_pcc_wo_all = compute_mse_decomposition(
        y_test, y_pred_without)
    
    print("\n  Model: CISIR (with wPCC)")
    mean_w_all, sd_w_all, corr_w_all, one_minus_pcc_w_all = compute_mse_decomposition(
        y_test, y_pred_with)
    
    results['all'] = {
        'without_wpcc': {'mean': mean_wo_all, 'sd': sd_wo_all, 'corr': corr_wo_all, 'one_minus_pcc': one_minus_pcc_wo_all},
        'with_wpcc': {'mean': mean_w_all, 'sd': sd_w_all, 'corr': corr_w_all, 'one_minus_pcc': one_minus_pcc_w_all}
    }
    
    # Compute decomposition for "Rare" subset (high delta_p events)
    print("\n" + "-"*80)
    print(f"MSE DECOMPOSITION - RARE TEST SAMPLES (n={n_rare}, δ ≥ {rare_threshold})")
    print("-"*80)
    
    if n_rare > 0:
        # Filter predictions and targets to rare samples only
        y_test_rare = y_test[rare_mask]
        y_pred_without_rare = y_pred_without[rare_mask]
        y_pred_with_rare = y_pred_with[rare_mask]
        
        print("\n  Model: CISIR w/o wPCC")
        mean_wo_rare, sd_wo_rare, corr_wo_rare, one_minus_pcc_wo_rare = compute_mse_decomposition(
            y_test_rare, y_pred_without_rare)
        
        print("\n  Model: CISIR (with wPCC)")
        mean_w_rare, sd_w_rare, corr_w_rare, one_minus_pcc_w_rare = compute_mse_decomposition(
            y_test_rare, y_pred_with_rare)
        
        results['rare'] = {
            'without_wpcc': {'mean': mean_wo_rare, 'sd': sd_wo_rare, 'corr': corr_wo_rare, 'one_minus_pcc': one_minus_pcc_wo_rare},
            'with_wpcc': {'mean': mean_w_rare, 'sd': sd_w_rare, 'corr': corr_w_rare, 'one_minus_pcc': one_minus_pcc_w_rare}
        }
    else:
        print("  ⚠ WARNING: No rare samples found in test set!")
        print(f"  No samples with δ ≥ {rare_threshold}")
    
    return results


def format_table_row(subset: str, without_wpcc: dict, with_wpcc: dict) -> str:
    """
    Format a row of the LaTeX table with deltas and percent changes.
    
    Computes the difference (Δ) and percent change (%Δ) for each MSE decomposition term
    and formats them for direct copy-paste into LaTeX table.
    
    Args:
        subset: Name of the subset ('All' or 'Rare')
        without_wpcc: Dict with keys 'mean', 'sd', 'corr', 'one_minus_pcc' for model without wPCC
        with_wpcc: Dict with keys 'mean', 'sd', 'corr', 'one_minus_pcc' for model with wPCC
    
    Returns:
        str: Formatted LaTeX table row string with ampersands and line break
    """
    # Compute deltas (Δ = with_wpcc - without_wpcc)
    mean_delta = with_wpcc['mean'] - without_wpcc['mean']
    sd_delta = with_wpcc['sd'] - without_wpcc['sd']
    corr_delta = with_wpcc['corr'] - without_wpcc['corr']
    one_minus_pcc_delta = with_wpcc['one_minus_pcc'] - without_wpcc['one_minus_pcc']
    
    # Compute percent changes (%Δ = Δ / without_wpcc * 100)
    mean_pct = (mean_delta / without_wpcc['mean'] * 100) if without_wpcc['mean'] != 0 else 0
    sd_pct = (sd_delta / without_wpcc['sd'] * 100) if without_wpcc['sd'] != 0 else 0
    corr_pct = (corr_delta / without_wpcc['corr'] * 100) if without_wpcc['corr'] != 0 else 0
    one_minus_pcc_pct = (one_minus_pcc_delta / without_wpcc['one_minus_pcc'] * 100) if without_wpcc['one_minus_pcc'] != 0 else 0
    
    # Format the LaTeX row with compact notation
    # Column structure: Subset | Term1: -wPCC, +wPCC, %Δ | Term2: -wPCC, +wPCC, %Δ | Term3: -wPCC, +wPCC, %Δ | 1-PCC: -wPCC, +wPCC, %Δ
    row = f"{subset:4s} & "
    # Mean term (3 columns)
    row += f"{without_wpcc['mean']:.3f} & {with_wpcc['mean']:.3f} & "
    row += f"\\textbf{{{mean_pct:+.1f}}} & "
    # SD term (3 columns)
    row += f"{without_wpcc['sd']:.3f} & {with_wpcc['sd']:.3f} & "
    row += f"\\textbf{{{sd_pct:+.1f}}} & "
    # Corr term (3 columns)
    row += f"{without_wpcc['corr']:.3f} & {with_wpcc['corr']:.3f} & "
    row += f"\\textbf{{{corr_pct:+.1f}}} & "
    # 1-PCC term (3 columns)
    row += f"{without_wpcc['one_minus_pcc']:.3f} & {with_wpcc['one_minus_pcc']:.3f} & "
    row += f"\\textbf{{{one_minus_pcc_pct:+.1f}}} \\\\"
    
    return row


def print_latex_table(results):
    """
    Print the formatted LaTeX table.
    
    Args:
        results: Dictionary with decomposition results
    """
    print("\n" + "="*80)
    print("LATEX TABLE")
    print("="*80)
    print()
    
    # Print "All" row
    if 'all' in results:
        print(format_table_row('All', results['all']['without_wpcc'], 
                              results['all']['with_wpcc']))
    
    # Print "Rare" row
    if 'rare' in results:
        print(format_table_row('Rare', results['rare']['without_wpcc'], 
                              results['rare']['with_wpcc']))
    
    print()


def find_model_paths(
    seed: int, 
    models_dir: str = "/home/paperspace/sep-forecasting-research/models"
) -> tuple[str, str]:
    """
    Find model file paths for a given seed.
    
    Searches for two models trained with the same seed:
    - CISIR w/o wPCC: Identified by 'quc_ssb' in filename (lambda=0)
    - CISIR: Identified by 'quc' without 'ssb' in filename (lambda>0)
    
    Args:
        seed: Random seed used during model training
        models_dir: Directory containing saved model files
    
    Returns:
        tuple: (path_without_wpcc, path_with_wpcc)
            - path_without_wpcc: Path to CISIR w/o wPCC model
            - path_with_wpcc: Path to CISIR model
    
    Raises:
        FileNotFoundError: If models for the given seed are not found
    """
    import glob
    
    # Pattern for model without wPCC (quc_ssb indicates lambda=0, no PCC regularization)
    pattern_without = f"{models_dir}/final_model_{seed}_*_quc_ssb_*_reg.h5"
    models_without = glob.glob(pattern_without)
    
    # Pattern for model with wPCC (quc without ssb indicates lambda>0, with PCC regularization)
    pattern_with = f"{models_dir}/final_model_{seed}_*_quc_*_reg.h5"
    models_with = [m for m in glob.glob(pattern_with) if 'quc_ssb' not in m]
    
    # Validate that models were found
    if not models_without:
        raise FileNotFoundError(
            f"No model without wPCC found for seed {seed}\n"
            f"  Searched pattern: {pattern_without}"
        )
    if not models_with:
        raise FileNotFoundError(
            f"No model with wPCC found for seed {seed}\n"
            f"  Searched pattern: {pattern_with} (excluding 'quc_ssb')"
        )
    
    # Take the first match (should be only one per seed)
    path_without_wpcc = models_without[0]
    path_with_wpcc = models_with[0]
    
    # Log the found paths
    print(f"  Seed {seed}:")
    print(f"    CISIR w/o wPCC: {os.path.basename(path_without_wpcc)}")
    print(f"    CISIR (w/ wPCC): {os.path.basename(path_with_wpcc)}")
    
    return path_without_wpcc, path_with_wpcc


def average_results(all_seed_results: list[dict]) -> dict:
    """
    Average MSE decomposition results across multiple random seeds.
    
    This provides more robust estimates by averaging over different random
    initializations and data splits.
    
    Args:
        all_seed_results: List of result dictionaries, one per seed
    
    Returns:
        dict: Averaged results with same structure as individual results
    """
    print(f"\n  Averaging results across {len(all_seed_results)} seeds...")
    
    # Initialize averaged results with zeros
    avg_results = {
        'all': {
            'without_wpcc': {'mean': 0.0, 'sd': 0.0, 'corr': 0.0, 'one_minus_pcc': 0.0},
            'with_wpcc': {'mean': 0.0, 'sd': 0.0, 'corr': 0.0, 'one_minus_pcc': 0.0}
        },
        'rare': {
            'without_wpcc': {'mean': 0.0, 'sd': 0.0, 'corr': 0.0, 'one_minus_pcc': 0.0},
            'with_wpcc': {'mean': 0.0, 'sd': 0.0, 'corr': 0.0, 'one_minus_pcc': 0.0}
        }
    }
    
    n_seeds = len(all_seed_results)
    
    # Sum up all values across seeds
    for result in all_seed_results:
        for subset in ['all', 'rare']:
            if subset in result:
                for model_type in ['without_wpcc', 'with_wpcc']:
                    for term in ['mean', 'sd', 'corr', 'one_minus_pcc']:
                        avg_results[subset][model_type][term] += result[subset][model_type][term]
    
    # Divide by number of seeds to compute mean
    for subset in ['all', 'rare']:
        for model_type in ['without_wpcc', 'with_wpcc']:
            for term in ['mean', 'sd', 'corr', 'one_minus_pcc']:
                avg_results[subset][model_type][term] /= n_seeds
    
    print(f"  ✓ Averaged complete")
    
    return avg_results


def print_detailed_results(seed, results):
    """Print detailed results for a single seed."""
    print(f"\n{'='*80}")
    print(f"SEED {seed} - DETAILED RESULTS")
    print(f"{'='*80}")
    
    for subset in ['all', 'rare']:
        if subset in results:
            print(f"\n{subset.upper()} subset:")
            print(f"  Without wPCC: mean={results[subset]['without_wpcc']['mean']:.6f}, "
                  f"sd={results[subset]['without_wpcc']['sd']:.6f}, "
                  f"corr={results[subset]['without_wpcc']['corr']:.6f}, "
                  f"1-PCC={results[subset]['without_wpcc']['one_minus_pcc']:.6f}")
            print(f"  With wPCC:    mean={results[subset]['with_wpcc']['mean']:.6f}, "
                  f"sd={results[subset]['with_wpcc']['sd']:.6f}, "
                  f"corr={results[subset]['with_wpcc']['corr']:.6f}, "
                  f"1-PCC={results[subset]['with_wpcc']['one_minus_pcc']:.6f}")


def main() -> dict:
    """
    Main function to compute MSE decomposition table across all seeds.
    
    This is the entry point that orchestrates:
    1. Loading the test dataset
    2. Finding model files for all seeds
    3. Computing MSE decomposition for each seed
    4. Averaging results across seeds
    5. Generating LaTeX tables and CSV outputs
    
    Returns:
        dict: Summary of all results including seeds processed, individual
              results, averaged results, and output file paths
    """
    print("\n" + "#"*80)
    print("# MSE DECOMPOSITION ANALYSIS: CISIR vs CISIR w/o wPCC")
    print("# " + "="*76)
    print("# Computing Murphy (1988) MSE decomposition terms for SEP-EC forecasting")
    print("#" + "#"*79)
    
    # Check GPU availability
    print("\n" + "="*80)
    print("GPU CONFIGURATION")
    print("="*80)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✓ GPU detected: {len(gpus)} device(s) available")
        for gpu in gpus:
            print(f"  - {gpu.name}")
        print("  Models will run on GPU for faster inference")
    else:
        print("⚠ WARNING: No GPU detected!")
        print("  Models will run on CPU (slower)")
        print("  Check your TensorFlow GPU installation")
    
    # Seeds to process (from globals.py TRIAL_SEEDS)
    seeds = TRIAL_SEEDS  # [456789, 42, 123, 0, 9999]
    print(f"\nProcessing {len(seeds)} random seeds: {seeds}")
    
    # Test data directory
    test_data_dir = '/home/paperspace/sep-forecasting-research/data/testing'
    
    # Load test dataset (same for all seeds, only predictions differ)
    print("\n" + "="*80)
    print("STEP 1: LOADING TEST DATASET")
    print("="*80)
    print(f"Data directory: {test_data_dir}")
    print(f"Inputs: {INPUTS_TO_USE[0]}")
    print(f"Outputs: {OUTPUTS_TO_USE}")
    
    X_test, y_test, _, _ = build_dataset(
        test_data_dir,
        inputs_to_use=INPUTS_TO_USE[0],
        add_slope=ADD_SLOPE[0],
        outputs_to_use=OUTPUTS_TO_USE,
        cme_speed_threshold=CME_SPEED_THRESHOLD[0]
    )
    print(f"✓ Test set loaded: X_test={X_test.shape}, y_test={y_test.shape}")
    
    # Find all model paths for each seed
    print("\n" + "="*80)
    print("STEP 2: FINDING MODEL FILES")
    print("="*80)
    print("Searching for CISIR and CISIR w/o wPCC models for each seed...")
    
    model_paths = {}
    for seed in seeds:
        try:
            model_paths[seed] = find_model_paths(seed)
        except FileNotFoundError as e:
            print(f"  ⚠ WARNING: {e}")
            print(f"  Skipping seed {seed}")
            continue
    
    if not model_paths:
        print("\n✗ ERROR: No models found for any seed!")
        print("  Please check that model files exist in the models directory.")
        return None
    
    print(f"\n✓ Found models for {len(model_paths)}/{len(seeds)} seeds")
    
    # Compute decomposition for each seed
    print("\n" + "="*80)
    print("STEP 3: COMPUTING MSE DECOMPOSITIONS FOR EACH SEED")
    print("="*80)
    
    all_seed_results = []
    for i, seed in enumerate(sorted(model_paths.keys()), 1):
        path_without, path_with = model_paths[seed]
        
        print(f"\n\n{'#'*80}")
        print(f"# SEED {i}/{len(model_paths)}: {seed}")
        print(f"{'#'*80}")
        
        # Compute MSE decomposition for this seed
        results = compute_table_values(
            path_without,
            path_with,
            X_test,
            y_test,
            rare_threshold=MAE_PLUS_THRESHOLD
        )
        
        all_seed_results.append(results)
        print_detailed_results(seed, results)
    
    # Average results across all seeds
    print("\n\n" + "="*80)
    print("STEP 4: AVERAGING RESULTS ACROSS ALL SEEDS")
    print("="*80)
    avg_results = average_results(all_seed_results)
    
    print_detailed_results("AVERAGE", avg_results)
    
    # Print LaTeX tables for each seed
    print("\n\n" + "="*80)
    print("LATEX TABLES - INDIVIDUAL SEEDS")
    print("="*80)
    for i, seed in enumerate(sorted(model_paths.keys())):
        print(f"\n--- Seed {seed} ---")
        print_latex_table(all_seed_results[i])
    
    # Print averaged LaTeX table (main result)
    print("\n\n" + "="*80)
    print("LATEX TABLE - AVERAGED ACROSS SEEDS (USE THIS FOR YOUR PAPER!)")
    print("="*80)
    print_latex_table(avg_results)
    
    # Save results to CSV files
    print("\n\n" + "="*80)
    print("STEP 5: SAVING RESULTS TO CSV FILES")
    print("="*80)
    
    # Create logs directory if it doesn't exist
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
        print(f"  Created directory: {logs_dir}/")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create detailed CSV with all seeds (one row per seed-subset-model combination)
    print("  Creating detailed CSV...")
    detailed_rows = []
    for i, seed in enumerate(sorted(model_paths.keys())):
        results = all_seed_results[i]
        for subset in ['all', 'rare']:
            if subset in results:
                detailed_rows.append({
                    'seed': seed,
                    'subset': subset,
                    'model': 'CISIR w/o wPCC',
                    'mean_term': results[subset]['without_wpcc']['mean'],
                    'sd_term': results[subset]['without_wpcc']['sd'],
                    'corr_term': results[subset]['without_wpcc']['corr'],
                    'one_minus_pcc': results[subset]['without_wpcc']['one_minus_pcc'],
                    'total_mse': (results[subset]['without_wpcc']['mean'] + 
                                 results[subset]['without_wpcc']['sd'] + 
                                 results[subset]['without_wpcc']['corr'])
                })
                detailed_rows.append({
                    'seed': seed,
                    'subset': subset,
                    'model': 'CISIR',
                    'mean_term': results[subset]['with_wpcc']['mean'],
                    'sd_term': results[subset]['with_wpcc']['sd'],
                    'corr_term': results[subset]['with_wpcc']['corr'],
                    'one_minus_pcc': results[subset]['with_wpcc']['one_minus_pcc'],
                    'total_mse': (results[subset]['with_wpcc']['mean'] + 
                                 results[subset]['with_wpcc']['sd'] + 
                                 results[subset]['with_wpcc']['corr'])
                })
    
    # Add averaged results
    for subset in ['all', 'rare']:
        if subset in avg_results:
            detailed_rows.append({
                'seed': 'AVERAGE',
                'subset': subset,
                'model': 'CISIR w/o wPCC',
                'mean_term': avg_results[subset]['without_wpcc']['mean'],
                'sd_term': avg_results[subset]['without_wpcc']['sd'],
                'corr_term': avg_results[subset]['without_wpcc']['corr'],
                'one_minus_pcc': avg_results[subset]['without_wpcc']['one_minus_pcc'],
                'total_mse': (avg_results[subset]['without_wpcc']['mean'] + 
                             avg_results[subset]['without_wpcc']['sd'] + 
                             avg_results[subset]['without_wpcc']['corr'])
            })
            detailed_rows.append({
                'seed': 'AVERAGE',
                'subset': subset,
                'model': 'CISIR',
                'mean_term': avg_results[subset]['with_wpcc']['mean'],
                'sd_term': avg_results[subset]['with_wpcc']['sd'],
                'corr_term': avg_results[subset]['with_wpcc']['corr'],
                'one_minus_pcc': avg_results[subset]['with_wpcc']['one_minus_pcc'],
                'total_mse': (avg_results[subset]['with_wpcc']['mean'] + 
                             avg_results[subset]['with_wpcc']['sd'] + 
                             avg_results[subset]['with_wpcc']['corr'])
            })
    
    detailed_df = pd.DataFrame(detailed_rows)
    detailed_csv = os.path.join(logs_dir, f"mse_decomposition_detailed_{timestamp}.csv")
    detailed_df.to_csv(detailed_csv, index=False)
    print(f"  ✓ Detailed CSV saved: {detailed_csv}")
    print(f"    ({len(detailed_rows)} rows: {len(model_paths)} seeds × 2 subsets × 2 models)")
    
    # Create summary CSV with deltas and percentages (for the LaTeX table)
    print("  Creating summary CSV (averaged results with Δ and %Δ)...")
    summary_rows = []
    for subset in ['all', 'rare']:
        if subset in avg_results:
            wo = avg_results[subset]['without_wpcc']
            wi = avg_results[subset]['with_wpcc']
            
            summary_rows.append({
                'subset': subset.capitalize(),
                # Mean term
                'mean_wo_wpcc': wo['mean'],
                'mean_w_wpcc': wi['mean'],
                'mean_delta': wi['mean'] - wo['mean'],
                'mean_pct_delta': ((wi['mean'] - wo['mean']) / wo['mean'] * 100) if wo['mean'] != 0 else 0,
                # SD term
                'sd_wo_wpcc': wo['sd'],
                'sd_w_wpcc': wi['sd'],
                'sd_delta': wi['sd'] - wo['sd'],
                'sd_pct_delta': ((wi['sd'] - wo['sd']) / wo['sd'] * 100) if wo['sd'] != 0 else 0,
                # Corr term
                'corr_wo_wpcc': wo['corr'],
                'corr_w_wpcc': wi['corr'],
                'corr_delta': wi['corr'] - wo['corr'],
                'corr_pct_delta': ((wi['corr'] - wo['corr']) / wo['corr'] * 100) if wo['corr'] != 0 else 0,
                # 1-PCC term
                'one_minus_pcc_wo_wpcc': wo['one_minus_pcc'],
                'one_minus_pcc_w_wpcc': wi['one_minus_pcc'],
                'one_minus_pcc_delta': wi['one_minus_pcc'] - wo['one_minus_pcc'],
                'one_minus_pcc_pct_delta': ((wi['one_minus_pcc'] - wo['one_minus_pcc']) / wo['one_minus_pcc'] * 100) if wo['one_minus_pcc'] != 0 else 0,
            })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(logs_dir, f"mse_decomposition_summary_{timestamp}.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"  ✓ Summary CSV saved: {summary_csv}")
    print(f"    (2 rows: All and Rare subsets)")
    
    # Create results summary dictionary
    results_summary = {
        'timestamp': timestamp,
        'seeds': sorted(list(model_paths.keys())),
        'n_seeds': len(model_paths),
        'averaged_results': avg_results,
        'rare_threshold': MAE_PLUS_THRESHOLD,
        'n_test_samples': len(y_test),
        'n_rare_samples': np.sum(y_test[:, 0] >= MAE_PLUS_THRESHOLD),
        'detailed_csv': detailed_csv,
        'summary_csv': summary_csv
    }
    
    # Print final summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"  Seeds processed: {results_summary['n_seeds']}")
    print(f"  Test samples: {results_summary['n_test_samples']}")
    print(f"  Rare samples: {results_summary['n_rare_samples']}")
    print(f"\nOutput files:")
    print(f"  1. {detailed_csv}")
    print(f"  2. {summary_csv}")
    print("\nUse the LaTeX table printed above for your paper!")
    
    return results_summary


if __name__ == '__main__':
    main()
