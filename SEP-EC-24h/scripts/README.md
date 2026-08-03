# Reproduction pipeline — SEP-EC-24h

Data files are not committed. Place these at the repo root first:

| File | What |
| --- | --- |
| `ephin5m_v2.dat` | raw EPHIN 5-min flux — **the v2 file**, the one with 2012 present |
| `curr_pf10th10_original.csv` | SEP event catalog (onset/peak/end, `Index` 1–4) |
| `SEP10MeV_Features_v2.csv` | CME features + `daily_sunspots` |
| `SN_d_tot_V2.0.csv` | `https://www.sidc.be/SILSO/DATA/SN_d_tot_V2.0.csv` |
| `full/` | the distributed SEP-EC `misc/full` (44 trimmed event files) |

Check you have the right flux file: 2012 must be ~72.4% valid, not 0%.
Two `ephin5m.dat` copies exist that both lack 2012 entirely.

## Run order

```bash
python build24_phase1.py    # regenerate 46 events, onset-24h .. end+24h -> rebuild24/
python build24_phase2.py    # splice pre + SEP-EC + post -> SEP-EC-24h/
python build24_validate.py  # 17-check gate; must report 0 FAIL
python build24_figs.py      # log10 and ln(1+I) figure sets
```

## Recovered parameters

The committed notebook does not reproduce the distributed dataset. Both of these were
established empirically and are baked into `build24_phase1.py`:

- `HOURS_BEFORE` was 12 in the original build (the notebook says 16; five events pin
  the start to onset − 12.00 h). Here it is 24, per the request.
- Channel prefixes are `p6.1` / `p33.0`, not the notebook's `e6.10` / `e33.00`.

## Note

`build24_phase1.py` reimplements `load_flux_data` with `sep=r"\s+"` and a vectorised
nearest-timestamp lookup, because `delim_whitespace` is deprecated and
`Index.get_loc(..., method='nearest')` was removed in pandas 2.x. Semantics are
unchanged from `modules/training/electron_ts.py`.
