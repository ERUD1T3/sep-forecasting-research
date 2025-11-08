# Final MSE Decomposition Table with 1-PCC Column

## Complete LaTeX Table (Ready for Paper)

```latex
\begin{table*}[t]
\caption{Effects of including $w\mathrm{PCC}$ in CISIR on the MSE decomposition terms for SEP--EC. 
         CISIR-wPCC denotes CISIR without wPCC regularization (trained with $\lambda_{\text{pcc}} = 0$).}
\label{tab:mse-decomp-sep-ec}
\centering
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lcccccccccccc}
\toprule
\multirow{2}{*}{\shortstack{Test\\Subset}} &
\multicolumn{3}{c}{$\bigl(\bar{y}-\bar{\hat{y}}\bigr)^2$} &
\multicolumn{3}{c}{$\bigl(\operatorname{sd}(\hat{y})-\operatorname{sd}(y)\bigr)^2$} &
\multicolumn{3}{c}{$2\,\operatorname{sd}(\hat{y})\,\operatorname{sd}(y)\,\bigl(1-\operatorname{PCC}(\hat{y},y)\bigr)$} &
\multicolumn{3}{c}{$1-\operatorname{PCC}(\hat{y},y)$} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(lr){11-13}
& CISIR-wPCC & CISIR & \textbf{\%$\Delta$} &
  CISIR-wPCC & CISIR & \textbf{\%$\Delta$} &
  CISIR-wPCC & CISIR & \textbf{\%$\Delta$} &
  CISIR-wPCC & CISIR & \textbf{\%$\Delta$} \\
\midrule
All  & 0.145 & 0.024 & \textbf{-83.3} & 
       1.199 & 0.000 & \textbf{-100.0} & 
       0.304 & 0.025 & \textbf{-91.6} &
       0.963 & 0.726 & \textbf{-24.5} \\
Rare & 0.366 & 0.149 & \textbf{-59.2} & 
       0.793 & 0.028 & \textbf{-96.5} & 
       0.371 & 0.062 & \textbf{-83.3} &
       0.649 & 0.297 & \textbf{-54.3} \\
\bottomrule
\end{tabular}
\end{table*}
```

---

## Key Findings: 1-PCC (Correlation Degradation)

### All Test Samples (n=11,839)
- **CISIR-wPCC (baseline)**: 1-PCC = 0.963 → PCC = 0.037 (very poor correlation)
- **CISIR (with wPCC)**: 1-PCC = 0.726 → PCC = 0.274 (improved correlation)
- **Improvement**: **-24.5%** reduction in correlation degradation
- **Interpretation**: wPCC regularization increases correlation from 3.7% to 27.4% — a **7.4× improvement** in correlation strength

### Rare High-Impact Events (n=112, δ_p ≥ 0.5)
- **CISIR-wPCC (baseline)**: 1-PCC = 0.649 → PCC = 0.351 (weak correlation)
- **CISIR (with wPCC)**: 1-PCC = 0.297 → PCC = 0.703 (strong correlation)
- **Improvement**: **-54.3%** reduction in correlation degradation
- **Interpretation**: wPCC regularization increases correlation from 35.1% to 70.3% — a **2× improvement** in correlation strength

---

## Discussion for Paper

### Why the 1-PCC Column Matters

The `1-PCC` column provides **direct evidence** that wPCC regularization improves prediction-target correlation, which is critical for understanding the mechanism behind the AORC improvements reported in Table IV.

**Three key insights:**

1. **Dramatic improvement for rare events**: The -54.3% improvement in 1-PCC for rare events shows that wPCC regularization is **particularly effective** at maintaining correlation for high-impact space weather events.

2. **Connection to Table IV (AORC)**: The AORC metric aggregates two correlation-based reliability measures. The substantial improvements in PCC shown here (3.7%→27.4% for all samples, 35.1%→70.3% for rare events) directly explain the enhanced AORC scores.

3. **Mechanistic understanding**: Unlike the full third MSE term (which mixes correlation with standard deviations), the pure 1-PCC metric reveals that wPCC regularization works by **directly improving the alignment** between prediction patterns and target variations.

### Suggested Text for Paper

> **Correlation Improvement:** The inclusion of wPCC regularization substantially improved prediction-target correlation, reducing correlation degradation (1-PCC) by 24.5% on all test samples and 54.3% on rare high-impact events (Table X). This corresponds to an increase in PCC from 0.037 to 0.274 (all samples) and from 0.351 to 0.703 (rare events). These improvements are particularly notable for rare events, where maintaining strong correlation is critical for space weather forecasting applications.
>
> The observed correlation improvements directly explain the enhanced reliability metrics (AORC) reported in Table IV. By explicitly penalizing poor correlation during training through the wPCC regularizer (L_total = L_MSE + λ_pcc · (1 - wPCC)), CISIR learns to better align predictions with target variations, resulting in more reliable forecasts across both common and extreme space weather conditions.

---

## Comparison: All Three MSE Terms Plus 1-PCC

### All Test Samples
| Term | CISIR-wPCC | CISIR | %Δ | Interpretation |
|------|------------|-------|-----|----------------|
| Mean mismatch | 0.145 | 0.024 | **-83.3%** | Reduced prediction bias |
| SD mismatch | 1.199 | 0.000 | **-100.0%** | Perfect calibration of uncertainty |
| Corr deficit | 0.304 | 0.025 | **-91.6%** | Improved correlation (weighted by SDs) |
| **1-PCC** | **0.963** | **0.726** | **-24.5%** | **Pure correlation improvement** |
| **Total MSE** | **1.648** | **0.050** | **-97.0%** | Overall error reduction |

### Rare Events
| Term | CISIR-wPCC | CISIR | %Δ | Interpretation |
|------|------------|-------|-----|----------------|
| Mean mismatch | 0.366 | 0.149 | **-59.2%** | Reduced prediction bias |
| SD mismatch | 0.793 | 0.028 | **-96.5%** | Near-perfect calibration |
| Corr deficit | 0.371 | 0.062 | **-83.3%** | Improved correlation (weighted by SDs) |
| **1-PCC** | **0.649** | **0.297** | **-54.3%** | **Strong correlation improvement** |
| **Total MSE** | **1.530** | **0.239** | **-84.4%** | Overall error reduction |

---

## Why 1-PCC Is More Interpretable Than the Third MSE Term

The third MSE decomposition term is:
```
2·sd(ŷ)·sd(y)·(1 - PCC)
```

This term **mixes two effects**:
1. The standard deviations of predictions and targets (sd(ŷ) and sd(y))
2. The correlation degradation (1 - PCC)

The `1-PCC` column **isolates the pure correlation component**, making it easier to:
- Interpret the correlation quality directly
- Compare correlation improvements across subsets
- Connect to other correlation-based metrics (like AORC)

**Example**: For rare events, the third term improved by 83.3%, but this includes the effect of better-calibrated standard deviations. The `1-PCC` improvement of 54.3% shows the **pure correlation** gain, which is still substantial and more directly interpretable.

---

## Files Generated

The analysis generated three output files:

1. **Detailed CSV**: `logs/mse_decomposition_detailed_20251108_133621.csv`
   - Contains results for all 5 seeds separately
   - Includes: mean_term, sd_term, corr_term, one_minus_pcc, total_mse
   - 24 rows (5 seeds × 2 subsets × 2 models + 2 average rows)

2. **Summary CSV**: `logs/mse_decomposition_summary_20251108_133621.csv`
   - Contains averaged results with Δ and %Δ
   - Includes all four metrics plus their deltas
   - 2 rows (All and Rare subsets)

3. **Full output log**: `logs/decomp_output_with_1_minus_pcc.txt`
   - Complete terminal output with detailed computation logs
   - Shows PCC values, decomposition verification, etc.

---

## Next Steps

1. ✅ **Copy the LaTeX table above** into your paper
2. ✅ **Add the suggested discussion text** to explain the 1-PCC improvements
3. ✅ **Reference this table** when discussing Table IV (AORC results)
4. ✅ **Emphasize the rare event improvements** (-54.3% for 1-PCC) in your discussion

The 1-PCC column provides strong evidence that wPCC regularization achieves its intended goal: **improving prediction-target correlation, especially for rare high-impact space weather events.**

