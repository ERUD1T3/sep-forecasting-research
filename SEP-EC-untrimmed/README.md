# SEP-EC (untrimmed + extended)

The SEP-EC electron/CME dataset with background restored **before** each event and
the decay extended **after** it.

## Why this exists

Two separate things shortened the distributed files (`misc/full/sep_event_*_filled_ie_trim.csv`):

1. **`trim_background()`** (`modules/training/utils.py`) dropped rows from the *start*
   of each event until proton intensity 3 h ahead exceeded 0.1 — discarding up to
   ~16 h of pre-onset background, and on 4 events cutting past onset into the rise.
2. **The window generator itself stopped at the catalog end time** (`Index 4` of
   `curr_pf10th10_original.csv`). Nothing was ever trimmed off the end — the series
   simply was never built past that point, so the decay is truncated well before
   background recovery.

Point 2 matters: **`_trim` did not remove the falling phase.** 41 of 44 distributed
events already contain a substantial fall. What was missing is the *tail* of it.

## Contents

| Path | What |
|---|---|
| `full/sep_event_N_filled_ie.csv` | 44 events, untrimmed + extended. Numbering matches SEP-EC. |
| `extra_catalog_events/` | 2 catalog events SEP-EC dropped (near-duplicates) |
| `manifest.csv` / `MANIFEST.md` | Per-event rows, restored hours, decay stats, provenance |

**40,855 rows** = 28,559 original + 4,562 restored before + 7,734 restored after
(**1,024.7 hours** restored: 380.2 h before across 39 events, 644.5 h after across 31).

## Schema

SEP-EC's 183 columns, unchanged, plus one appended column:

| `is_reconstructed` | Meaning |
|---|---|
| `0` | Original SEP-EC row, byte-identical |
| `1` | Restored pre-onset background |
| `2` | Restored post-event background |

Every file is ordered `1* 0+ 2*` — restored-before, original, restored-after — with no
interleaving, verified on all 44.

## What the extension buys

| | Distributed SEP-EC | This dataset |
| --- | --- | --- |
| Median decay captured after peak | 0.79 decades / 19.3 h | **1.48 decades / 34.3 h** |
| Events with ≥1 decade of decay | 17 / 44 | **30 / 44** |
| Events with ≥2 decades | — | 12 / 44 |
| Events that look rise-only (peak in last 10%) | 4 / 44 | **2 / 44** (19, 30) |

## Guarantees, and how they were checked

**Original rows are untouched.** Every `is_reconstructed=0` row is byte-identical to
the distributed file — verified as *raw text*, all 28,559 lines across all 44 events,
preserving the native `M/D/YYYY H:MM` timestamps and exact float representations.
Filtering `is_reconstructed == 0` reproduces `misc/full` exactly.

Also verified 44/44: every SEP-EC timestamp present, time monotonic, contiguous
5-minute grid, correct block ordering.

**Hold-out test on the reconstruction.** Five events were never trimmed, so SEP-EC
contains ground truth for exactly the kind of deep pre-onset rows that are
reconstructed elsewhere. Comparing the reconstruction against it (first 150 rows):

| Event | Era | Flux agreement | Proton Intensity |
| --- | --- | --- | --- |
| 7, 38, 39 | raw flux | **100.0000%** | 100% |
| 12 | 2012 | **100.0000%** | 100% |
| 11 | 2012 | 92.51% | **100%** |

Non-2012 reconstruction is bit-exact, and **`Proton Intensity` is exact in all five**.
Event 11's shortfall is confined to 23 contiguous rows at the very start and only to
lag columns (worst at `tminus24`, decaying to `tminus23`); no `_t` column and no
`Proton Intensity` value is affected.

Regeneration measured against the full distributed dataset: **99.73%** cell agreement,
25/44 events bit-exact, 100% row coverage.

**Post-event rows are real measurements, not filler.** Three cuts are applied, in order:

1. **Next catalog onset** — the window never extends into the following SEP event.
   Without this, event 10 ran straight into event 11 (onset 2.7 h after event 10's end)
   and its "background" rebounded 152× above its running minimum.
2. **Rebound guard** — stop where intensity climbs >3× above its running minimum,
   catching a shock arrival or an event the catalog missed.
3. **Raw coverage** — keep only while cumulative real-data coverage stays ≥80%.

After these, no event rebounds more than 2.99×, median backing is 100% (29 of 31
events fully backed, minimum 80.1%), there are zero `-9999` sentinels and no
extrapolated flat tails. `manifest.csv` reports `post_raw_backed_pct` per event.

**Restored-before rows are not held to the same bar.** The pre-onset rows reproduce
what the *original* pipeline would have built (its `interpolate_and_extrapolate` step
included), so no coverage filter is applied — dropping rows there would make the file
less faithful to the untrimmed original, not more. They are 95.3% raw-backed overall
and 35 of 39 events are at 100%, but three fall short and `manifest.csv` reports
`pre_raw_backed_pct` per event so they can be filtered:

| Event | Pre rows | Raw-backed |
| --- | --- | --- |
| 31 | 70 | **0.0%** — a single constant, extrapolated |
| 30 | 121 | 37.2% |
| 43 | 116 | 45.7% |

Event 31's pre-onset segment is one repeated value; treat it as a placeholder, not a
measurement.

**How far the decay actually gets.** Median levels across the 31 extended events:
peak 3.43 → 0.481 at the start of the post window → **0.074 at the end**, against a
median pre-onset background of 0.0137. So 18/31 events end below 0.1 and 26/31 below
0.3, but only 13/31 land within 3× of their own pre-event background.

**Call this extended decay, not background recovery.** SEP decay runs for days; 24 h
gets most events close to quiet levels but not all the way down.

## Missing values

There are **none**. Across 7,394,755 numeric cells: zero NaN, zero `-9999` sentinels
(the pipeline's "no data available" marker), zero blank fields. The 5-minute grid is
contiguous with no time gaps in all 44 events.

Two things that look like missing data but are not:

- **Zeros in the flux columns** (32,056 cells, 0.5%) are genuine non-detections — the
  instrument reading zero, not an absent value. The plots draw them as line gaps
  rather than clamping them to a floor.
- **Zeros in the CME/context columns** (58.5% of those cells) mean *no CME is active*
  for that timestamp; `cme_donki_time` is likewise the literal `0`. That is the
  original pipeline's default, not a gap.

What *is* worth knowing is that some values are interpolated rather than measured —
see `pre_raw_backed_pct` / `post_raw_backed_pct` in the manifest, and note that the
distributed 2012 events carry their own interpolation (events 14, 17 and 18 contain
runs of >20 identical consecutive values, up to 61 in event 17).

## Limitations — read before drawing conclusions

- **13 events get no post-event extension**: 7, 11–20, 22, 23. Most are in the 2012
  flux blackout (below); event 7's post window is only ~44% covered in the raw file,
  below the 80% bar; events 17 and 30 are cut almost immediately by the next SEP onset.
- **Three events get only a short extension** because the next event starts soon after:
  32 (0.6 h), 30 (2.0 h), 10 (2.6 h).
- **The 2012 restored rows are not independent measurements.** The surviving
  `ephin5m.dat` has a total blackout from Dec 2011 to Jan 2013 (0 valid rows in all of
  2012; adjacent years are 81–96% complete) — that is the one substantive thing the
  lost `ephin5m_v2.dat` had. For those events the flux was recovered by inverting the
  derived datasets back into a time→flux table, so those values faithfully continue
  the dataset but **cannot be cited as evidence of what the instrument recorded**.
  Rows are marked `era=2012-blackout` in the manifest.
- **A triangular wedge of deep-lag cells** at the start of 2012 events is interpolated
  rather than sourced (<1% of the dataset). `Proton Intensity` is unaffected.
- **Restored rows cannot be validated directly** — no ground truth exists for them.
  The hold-out above covers 5 events; trust elsewhere rests on the same code and the
  same raw flux behaving identically, which is strong but is not direct measurement.
- **The extension is at most 24 h past the catalog end**, not "until background
  recovery" — and it is cut earlier by the next onset, a rebound, or sparse coverage.
  Slow events remain elevated at the end of the window.

## Reproducing

See `scripts/`. Data files are not committed. Two pipeline parameters had to be
recovered empirically because the committed notebook does not reproduce the shipped
data: `hours_before = 12` (the notebook says 16; five events pin the start to
onset − 12.00 h) and channel prefixes `p6.1` / `p33.0` (the notebook emits
`e6.10` / `e33.00`).

## Note on the log transform

The stored features are raw (pre-log) and `delta_log_Intensity` is
`log1p(Proton Intensity) − log1p(p_t)` (post-log). But the loader closes that gap:
`load_file_data(apply_log=True)` — the default in `modules/training/ts_modeling.py` —
applies `np.log1p` to the input columns and to `Proton Intensity` at load time. In
training, features and target are both in log space; only the on-disk representation
differs.
