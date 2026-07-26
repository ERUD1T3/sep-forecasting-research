# SEP-EC (untrimmed)

The SEP-EC electron/CME dataset with the pre-onset background **restored**.

The distributed SEP-EC files (`misc/full/sep_event_*_filled_ie_trim.csv`) had their
leading rows removed by `trim_background()` (`modules/training/utils.py`), which
drops rows from the start of each event until the proton intensity 3 hours ahead
exceeds 0.1. That discarded up to ~16 h of pre-onset background per event — and on
4 events it cut past onset into the rising phase. This dataset puts it back.

## Contents

| Path | What |
|---|---|
| `full/sep_event_N_filled_ie.csv` | 44 events, untrimmed. Numbering matches SEP-EC. |
| `extra_catalog_events/` | 2 catalog events that SEP-EC dropped (near-duplicates) |
| `manifest.csv` | Per-event row counts, restored hours, provenance |

**33,121 rows** total = 28,559 original + **4,562 restored** (380.2 hours across 39 events).

## Schema

Identical to SEP-EC's 183 columns, plus one appended column:

- **`is_reconstructed`** — `0` = row came from the distributed SEP-EC file, `1` = restored background.

## Guarantee on the original rows

Every `is_reconstructed=0` row is **byte-identical** to the distributed SEP-EC file —
verified as raw text across all 28,559 lines in all 44 events, including the native
`M/D/YYYY H:MM` timestamp format and exact float representations. Nothing that
already existed was recomputed or re-formatted.

Verified: 44/44 events monotonic in time, 44/44 on a contiguous 5-minute grid,
39/39 splice seams continuous (5-min spacing, <1 log-unit intensity step),
zero NaN and zero `-9999` sentinels in restored rows.

## How the restored rows were produced

Regenerated with the original pipeline (`notebooks/building_electron_ts_dataset.ipynb`
+ `modules/training/`), from `curr_pf10th10_original.csv`, `SEP10MeV_Features_v2.csv`,
`SN_d_tot_V2.0.csv` and the raw EPHIN flux. Two parameters were recovered empirically
because the committed notebook does not match the shipped data:

- `hours_before = 12` (the notebook says 16; 5 shipped events pin the start to onset−12 h exactly)
- channel prefixes `p6.1` / `p33.0` (the notebook emits `e6.10` / `e33.00`)

Validation of the regeneration against SEP-EC: **99.73%** cell agreement over all 44
events, 25/44 bit-exact, with 100% row coverage.

### The 2012 caveat

The surviving raw flux file `ephin5m.dat` has a **total blackout from Dec 2011 to
Jan 2013** (0 valid rows in all of 2012; adjacent years are 81–96% complete). The lost
`ephin5m_v2.dat` had it filled — that was the only substantive v1→v2 difference.

For the 14 events in 2012, the flux was recovered by inverting the derived datasets
back into a time→flux table (each event file stores flux at `Timestamp` plus 24 lags),
sourced from `misc/full` and archived git snapshots. This lifted 2012 from 0% to
**99.88%** agreement. `manifest.csv` marks these rows `restored_source=patched-2012`;
all other restored rows are `raw-v1`, straight from the raw flux file.

Residual (~0.3%) is confined to the deepest lag columns of the first ~23 rows of 2012
events, and is a hard ceiling: `interpolate_and_extrapolate` runs per column, so a
timestamp missing from the raw flux receives a different filled value in each of the
25 lag columns, and no single patched value can reproduce all of them. Only genuine
2012 raw flux — the lost `ephin5m_v2.dat` or a fresh EPHIN download — closes it.

## Known limitation

This restores background **before** onset. It does **not** extend past each event's
catalog end time (`Index 4`), so the post-event decay tail is still cut where SEP-EC
cut it — that data never existed in any version. Extending it is a one-parameter
change (`end_time` in `build_time_grid`) in `scripts/rebuild_untrimmed.py`, re-run
against the raw flux.
