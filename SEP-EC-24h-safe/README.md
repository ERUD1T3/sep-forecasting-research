# SEP-EC-24h (safe)

24 hours of background before onset and after the event end, **but only where the
instrument actually has data and no neighbouring event intrudes**.

Use this version when you need every added row to be a real measurement.
See `SEP-EC-24h-exact/` for the variant that is always exactly 24 h.

## Numbers

| | |
| --- | --- |
| Rows | **47,927** = 28,559 distributed + 9,045 before + 10,323 after |
| Full 24 h before onset | **34/44** |
| Full 24 h after end | **34/44** |
| Added rows backed by real flux | median **100.0%**, min **80.2%** |
| Events with added rows under 80% real | **0** |
| Windows reaching into the previous event | **1** |
| Windows reaching into the next event | **0** |

## Why 15 events fall short of 24 h

Four guards stop a window early, in this order:

1. **Previous-event guard** — the pre-window never crosses the preceding event's end.
2. **Next-onset guard** — the post-window never enters the following SEP event.
3. **Rebound guard** — stops where intensity climbs >3× above its running minimum,
   but only while still >10× the event's own background, so ordinary background
   fluctuation is not mistaken for a new event.
4. **Coverage rule** — added rows are kept only while cumulative real-data coverage
   stays ≥80%, so a sparse region cannot become a long interpolated tail.

Almost all shortfalls are the coverage rule: EPHIN simply has gaps there (SOHO antenna
keyhole/roll maneuvers), and 2012 is worst because v2 covers only 72.4% of that year.
Three events stop because a new event begins — event 10's post-window would otherwise
contain event 11 rising **118×**, and event 30's a **96×** rise.

`manifest.csv` names the limiter and the real-data percentage for every event.

## Caveats that apply to both

- **"24 h of background" is not always flat.** Measured over the pre-onset blocks,
  most are quiet but **6 swing more than a decade** — a real
  preceding disturbance, not an artifact. `pre_swing_decades` in the manifest.
- **24 h after the end is extended decay, not recovery.** SEP decay runs for days;
  most events are still above their pre-event background at the end of the window.
- **2012 has 72.4% raw coverage**, so 2012 windows contain interpolated stretches.
  That is the instrument record, not a pipeline gap.
- **Added rows cannot be validated directly** — no ground truth exists for them. A
  hold-out on the five never-trimmed events shows non-2012 reconstruction is bit-exact
  and the target is exact in all five.

## Validation

`scripts/build24_validate.py safe` — **0 FAIL / 16 checks**, including: every column
byte-identical in all distributed rows (44/44); strict-superset raw-text test; CME
attribute columns zero in every added row; contiguous 5-minute grid; one identical
184-column header; no NaN, no `-9999`; `p16.4_tplus6[i] == p16.4_t[i+6]`; column D and
EE–FD renamed as specified.

## Both versions share

- Regenerated from the recovered **`ephin5m_v2.dat`** — the flux file that actually
  contains 2012. The previously used `ephin5m.dat` has **zero** valid data for all of
  calendar 2012; v2 has 72.42%, matching the Kiel COSTEP/EPHIN archive's own holding
  for that year. Every other year is equivalent, so 2012 was v2's only real difference.
  Those rows are now independent measurements.
- **Strict superset of the distributed dataset.** Filtering `is_reconstructed == 0`
  reproduces `misc/full` **byte-for-byte** — all 28,559 lines, all 44 events, verified
  as raw text including timestamp format and float representation.
- **Column naming (1c):** column D is **`p16.4_tplus6`**; columns EE–FD are
  **`p16.4_tminus24 … p16.4_max_intensity`**. `p16.4` is the EPHIN 16.40 MeV proton
  channel — previously the only channel without its energy in the name. `t+6` = 6 steps
  × 5 min = the **+30 min** target offset, which is also the pipeline's own internal
  name for it (`target_tplus6`).
- **CME features (1d):** the 17 CME attribute columns are **`0` in every added row**.
  Inside the event they are untouched, so they still begin 30 minutes after the CME
  time and end with the event exactly as before. The four rolling counters
  (`CMEs Past Month`, `CMEs Past 9 Hours`, `CMEs Speed > 1000`, `Max CME Speed`) are
  left computed everywhere — they describe CME history at a timestamp, not an active
  CME, and zeroing them would assert "no CMEs in the past month", which is false.
- `is_reconstructed`: `0` = distributed row, `1` = added before onset, `2` = added
  after event end. Always ordered `1* 0+ 2*`.
- Schema is 184 columns, identical across all 46 files in the folder.

## Parser notes

Timestamps are `M/D/YYYY H:MM`, **unpadded** (`6/16/2012 0:09`) — parse with
`%m/%d/%Y %H:%M`. `Target Timestamp` is always exactly `Timestamp + 30 minutes`.
`cme_donki_time` is a union: literal `0` (no CME) or a timestamp in the same format.
Every other column is numeric. Zeros in CME columns mean *no CME active*, so
`CME_DONKI_speed = 0` is not a CME with zero speed.
