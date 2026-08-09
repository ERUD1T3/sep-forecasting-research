# SEP-C unnormalized, row-aligned

The SEP-C features in physical units, with timestamps, **aligned row-for-row to the
distributed SEP-C files** so predictions can be plotted against real feature values.

## Why not just use `misc/sep_10mev_full_raw.csv`

That file already exists in the SEP-C distribution and already has raw values and
timestamps — but it is in **source order**, while predictions come out of
`sep_10mev_training.csv` / `sep_10mev_testing.csv` / the fold files, which are
stratified, split and reordered. Nothing in those files links a row back to the raw
table, so joining prediction *i* to a raw speed is guesswork.

These files fix that: one raw file per distributed file, **same row count and same row
order**, so prediction *i* corresponds to row *i*.

## Files

| Output | Aligned to | Rows |
| --- | --- | --- |
| `sep_10mev_training_raw.csv` | `sep_10mev_training.csv` | 1,531 |
| `sep_10mev_testing_raw.csv` | `sep_10mev_testing.csv` | 766 |
| `fold{0..3}_sep_10mev_subtraining_raw.csv` | `fold{N}/sep_10mev_subtraining.csv` | 1,149 each |
| `fold{0..3}_sep_10mev_validation_raw.csv` | `fold{N}/sep_10mev_validation.csv` | 383 each |
| `misc_sep_10mev_full_raw.csv` | `misc/sep_10mev_full.csv` | 2,297 |
| `alignment_report.csv` | per-file verification | — |

## Columns

`source_row` (index into `SEP10MeV_Features_v2.csv`), then the three timestamps, then
the 22 features in physical units, then `peak_intensity` and `ln_peak_intensity`.

Features are raw, not normalized: `CME_DONKI_speed` runs **60 to 2800 km/s**, so a
1000 km/s CME reads as 1000.

## Timestamp availability

| Column | Present on |
| --- | --- |
| `CME_DONKI_time` | **2,297 / 2,297** |
| `CME_CDAW_time` | **2,297 / 2,297** |
| `SEP_onset_time` | **83 / 2,297** |

`SEP_onset_time` exists only for the 83 rows that are actual SEP events. The other
2,214 rows are CMEs that produced no SEP — their `ln_peak_intensity` is the floor
value (-1.609438 = ln 0.2). **Use `CME_DONKI_time` as the general lookup key.**

## How the alignment was recovered, and verified

The normalization in `ds_prep_sep_cme.ipynb :: preprocess_cme_features` was reproduced
exactly from `SEP10MeV_Features_v2.csv` (min-max per column, plus the three
log-transformed features). Each distributed row was then matched back to its source row
by the fingerprint of all 23 normalized columns.

Result across all 11 files: **every row matched, zero ambiguous matches.** As an
independent check, the selected source rows were re-normalized and compared against the
distributed values — identical to within 1e-9 on every file. See
`alignment_report.csv`.

Spot check: distributed `CME_DONKI_speed_norm = 0.525547` → 60 + 0.525547 × (2800 − 60)
= **1500 km/s**, which is what the raw file shows for that row. `ln_peak_intensity`
matches row-for-row in every file.

## Reproducing

`scripts/build_sepc_raw.py`. Needs `SEP10MeV_Features_v2.csv` at the repo root and the
distributed SEP-C folder (defaults to `~/Downloads/CISIR-data/SEP-C`).
