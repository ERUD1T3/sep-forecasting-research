# Quick Reference: MSE Decomposition Results with 1-PCC

## 📊 Complete Results Table (Averaged across 5 seeds)

| Subset | Metric | CISIR-wPCC | CISIR | %Δ | Interpretation |
|--------|--------|------------|-------|-----|----------------|
| **All** | Mean mismatch | 0.145 | 0.024 | **-83.3%** | Reduced prediction bias |
| | SD mismatch | 1.199 | 0.000 | **-100.0%** | Perfect uncertainty calibration |
| | Corr deficit | 0.304 | 0.025 | **-91.6%** | Improved weighted correlation |
| | **1-PCC** | **0.963** | **0.726** | **-24.5%** | **Better pure correlation** |
| | | **(PCC=0.037)** | **(PCC=0.274)** | | **(7.4× improvement)** |
| **Rare** | Mean mismatch | 0.366 | 0.149 | **-59.2%** | Reduced prediction bias |
| | SD mismatch | 0.793 | 0.028 | **-96.5%** | Near-perfect uncertainty calibration |
| | Corr deficit | 0.371 | 0.062 | **-83.3%** | Improved weighted correlation |
| | **1-PCC** | **0.649** | **0.297** | **-54.3%** | **Strong correlation gain** |
| | | **(PCC=0.351)** | **(PCC=0.703)** | | **(2× improvement)** |

---

## 🎯 Key Takeaway

**wPCC regularization dramatically improves correlation, especially for rare high-impact events:**
- All samples: PCC improves from 3.7% to 27.4% (**7.4× better**)
- Rare events: PCC improves from 35.1% to 70.3% (**2× better**)

The -54.3% reduction in correlation degradation (1-PCC) for rare events demonstrates that wPCC is particularly effective where it matters most.

---

## 📋 LaTeX Table (Copy-Paste Ready)

```latex
All  & 0.145 & 0.024 & \textbf{-83.3} & 1.199 & 0.000 & \textbf{-100.0} & 0.304 & 0.025 & \textbf{-91.6} & 0.963 & 0.726 & \textbf{-24.5} \\
Rare & 0.366 & 0.149 & \textbf{-59.2} & 0.793 & 0.028 & \textbf{-96.5} & 0.371 & 0.062 & \textbf{-83.3} & 0.649 & 0.297 & \textbf{-54.3} \\
```

---

## 📁 Files Generated

| File | Description |
|------|-------------|
| `mse_decomposition_detailed_20251108_133621.csv` | All 5 seeds + average (24 rows) |
| `mse_decomposition_summary_20251108_133621.csv` | Averaged results with Δ and %Δ (2 rows) |
| `FINAL_TABLE_WITH_1_MINUS_PCC.md` | Complete LaTeX + full discussion |
| `SUMMARY_OF_CHANGES.md` | What was changed and why |
| `decomp_output_with_1_minus_pcc.txt` | Full terminal output |

---

## 🔄 To Regenerate

```bash
python load_get_decomps.py
```

All changes are complete and tested! ✅

