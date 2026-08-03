# SEP-EC-24h

The SEP-EC electron/CME dataset extended to **24 hours before onset** and **24 hours
after the catalog event end**, regenerated from the recovered raw flux file.

Supersedes `SEP-EC-untrimmed/`, which used a 12 h pre-window and a flux file missing
all of 2012.

## What changed since the previous version

**The real `ephin5m_v2.dat` was found.** The earlier build used a copy of
`ephin5m.dat` with **zero valid data for all of calendar 2012**; the 2012 flux had to
be reconstructed by inverting SEP-EC's own lag columns, which was faithful but
circular. The recovered v2 file has **72.42%** coverage for 2012 — matching the Kiel
COSTEP/EPHIN archive's own holding for that year almost exactly. All other years are
identical across both files, so 2012 was v2's only substantive difference.

Consequences: the 2012 rows are now **independent measurements**, and 10 of the 14
2012 events that previously could get no post-event extension now can.

## Contents

| Path | What |
| --- | --- |
| `full/sep_event_N_24h.csv` | 44 events. Numbering matches SEP-EC. |
| `extra_catalog_events/` | 2 catalog events SEP-EC dropped (near-duplicates), same schema |
| `manifest.csv` / `MANIFEST.md` | Per-event hours, limiters, raw-backing, CME changelog |
| `cme_changelog.csv` | Which rows had CME columns recomputed |
| `figures/` | All 44 events in log10 flux space and in the network's `ln(1+I)` |
| `scripts/` | Full regeneration + validation chain |

**47,927 rows** = 28,559 distributed + 9,045 before + 10,323 after.
**902 h restored before onset, 861 h after** — median **24.0 h on both sides**.
Full 24 h achieved on **34/44 events before** and **34/44 after**; `manifest.csv`
gives the reason for every shortfall.

## Column naming (per your 1c)

| Old | New |
| --- | --- |
| `Proton Intensity` (column D) | **`p16.4_tplus6`** |
| `p_tminus24` … `p_max_intensity` (EE–FD) | **`p16.4_tminus24` … `p16.4_max_intensity`** |

`p16.4` is the EPHIN 16.40 MeV proton channel — previously the only channel without
its energy in the name. `t+6` marks the target as +6 steps × 5 min = **+30 min** ahead,
which is also the pipeline's own internal name for it (`target_tplus6`). The schema now
reads end to end as `p16.4_tminus24 … p16.4_t, p16.4_tplus6`.

`is_reconstructed`: `0` = distributed SEP-EC row, `1` = restored before onset,
`2` = restored after event end. Blocks are always ordered `1* 0+ 2*`.

## CME features (per your 1d)

The rule is unchanged and is what the pipeline always did: a CME becomes visible
**30 minutes after its DONKI time** and its attributes persist for
`avg_sep_duration` = **20.23 h**.

What *did* change: the original code only considered CMEs whose start time fell
**inside the window** (`features_t[0] <= cme_time <= features_t[-1]`). With a 12 h
window, a CME that began earlier but was still within its 20.23 h of persistence was
invisible. Widening to 24 h makes some of those visible, so the rule now runs over the
full window — otherwise CME features would abruptly drop to zero mid-persistence at
the splice boundary.

**This is the only change to rows you already had:** 7,992 cells across
**9 events and 526 rows** (1.65% of CME cells, 1.8% of rows). Per-event detail in
`cme_changelog.csv`. Everything else in those rows — all flux and lag columns, the
target, `delta_log_Intensity`, `Sunspot Number`, the four CME-history counters,
timestamps — is **byte-identical** to the distributed files.

## Honest caveats

- **"24 h of background" is not always flat.** Measured over the restored pre-onset
  blocks: 26 of 39 swing less than 0.5 decades (genuinely quiet), but **6 contain a
  >1-decade excursion** — a real preceding disturbance, not an artifact (events 33, 15,
  6, 28, 2, 24). Same on the post side: 26 of 39 quiet, 4 disturbed.
- **After 24 h the decay has not reached background** for most events. Median decay
  captured after peak is now 1.69 decades over 35.2 h (34/44 events reach ≥1 decade,
  16/44 reach ≥2), but SEP decay runs for days. Call it extended decay, not recovery.
- **10 events fall short of 24 h before and 10 short of 24 h after** — almost all
  because the raw flux there is under the 80% coverage bar, not by choice.
  `manifest.csv` names the limiter per event.
- **2012 has 72.4% raw coverage**, so 2012 windows contain interpolated stretches. This
  is the instrument record, not a pipeline gap.
- **Restored rows cannot be validated directly** — no ground truth exists for them. A
  hold-out on the five never-trimmed events shows non-2012 reconstruction is bit-exact
  and the target is exact in all five.

## Guards on the restored data

- **Previous-event guard** — the pre-window never crosses the preceding catalog event's end.
- **Next-onset guard** — the post-window never extends into the following SEP event.
- **Rebound guard** — stops where intensity climbs >3× above its running minimum,
  but only while still >10× the event's own background, so ordinary background
  fluctuation is not mistaken for a new event.
- **Coverage rule** — restored rows are kept only while cumulative real-data coverage
  stays ≥80%, so a sparse region cannot produce a long interpolated tail.

## Validation

`scripts/build24_validate.py` — **0 FAIL / 17 checks**:

- non-CME columns byte-identical in every distributed row (44/44)
- every SEP-EC timestamp present; time monotonic; contiguous 5-minute grid (44/44)
- `is_reconstructed` ordered `1*0+2*` (44/44); no CME snapping to zero at a seam
- one identical 184-column header across all 46 files; UTF-8, LF, no BOM, no quoting,
  no ragged rows, no empty or padded fields
- no NaN, no `-9999`, no infinities
- `p16.4_tplus6[i] == p16.4_t[i+6]`; lag self-consistency; `delta_log_Intensity` formula
- column D and EE–FD renamed as specified; no bare `p_` columns remain

## Parser notes

Timestamps are `M/D/YYYY H:MM`, **unpadded** (`6/16/2012 0:09`) — parse with
`%m/%d/%Y %H:%M`. `Target Timestamp` is always exactly `Timestamp + 30 minutes`.
`cme_donki_time` is a union: the literal `0` (no CME active) or a timestamp in the same
format. Every other column is numeric. Zeros in the CME columns mean *no CME active*,
so `CME_DONKI_speed = 0` is not a CME with zero speed.
