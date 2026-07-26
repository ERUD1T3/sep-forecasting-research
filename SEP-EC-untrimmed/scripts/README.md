# Reproduction pipeline

Regenerates the untrimmed SEP-EC dataset. Data files are **not** committed (see the
repo's `*.csv` / `.dat` ignore rules); place the inputs at the repo root first.

## Inputs

| File | Where to get it |
| --- | --- |
| `ephin5m.dat` | raw EPHIN 5-min flux (the surviving v1; note the 2012 blackout) |
| `curr_pf10th10_original.csv` | SEP event catalog (onset/peak/end, `Index` 1–4) |
| `SEP10MeV_Features_v2.csv` | CME features + `daily_sunspots` |
| `SN_d_tot_V2.0.csv` | `https://www.sidc.be/SILSO/DATA/SN_d_tot_V2.0.csv` |
| `full/` | the distributed SEP-EC `misc/full` (44 trimmed event files) |

## Run order

```bash
python rebuild_untrimmed.py   # 1. regenerate all 46 catalog events -> rebuild/
python harvest_mode.py        # 2. invert 2012 flux out of the derived datasets
python patch_rerun.py         # 3. patch the 2012 blackout, re-run those events
python validate.py            # 4. diff the regeneration against SEP-EC
python rebuild_extended.py    # 5. regenerate with the window extended past event end
python make_final.py          # 6. splice: pre + SEP-EC verbatim + post
```

Step 5 (`HOURS_AFTER = 24`) also records a per-row `flux_raw_backed` mask taken
*before* interpolation. Step 6 uses it to emit post-event rows only while cumulative
raw coverage stays ≥ `MIN_BACKED` (0.8), so a sparse region cannot produce a long
interpolated tail. Events failing that bar get no post rows — the 2012 blackout
events, plus event 7 (~44% covered).

Step 2 writes `harvest2012_mode.pkl`; step 3 reads a pickle of the same structure.
The variant that produced the shipped result used shipped-priority consolidation —
if you re-run, point `patch_rerun.py` at whichever pickle you generated and confirm
with `validate.py` (expected: ~99.73% cell agreement over all 44 events).

## Two recovered parameters

The committed notebook does **not** reproduce the distributed dataset. Both of these
were established empirically and are baked into `rebuild_untrimmed.py`:

- `HOURS_BEFORE = 12` — the notebook says 16, but 5 shipped events start at exactly
  onset − 12.00 h and none exceed 12 h.
- Channel prefixes `p6.1` / `p33.0` — the notebook's `channel_suffix` logic emits
  `e6.10` / `e33.00`, which no shipped file uses.

## Note

`rebuild_untrimmed.py` reimplements `load_flux_data` with `sep=r"\s+"` and a
vectorised nearest-timestamp lookup, because `delim_whitespace` is deprecated and
`Index.get_loc(..., method='nearest')` was removed in pandas 2.x. Semantics are
unchanged from `modules/training/electron_ts.py`.
