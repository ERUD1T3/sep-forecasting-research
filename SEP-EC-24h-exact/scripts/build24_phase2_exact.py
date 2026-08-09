"""
Phase 2 (EXACT variant) — splice with the window forced to exactly
onset-24h .. catalog_end+24h for every event. No guards at all: gaps are filled by
the pipeline's own interpolation, and a neighbouring event inside the window is kept.

Phase 2 — splice into the deliverable.

Per event:  [ pre-onset (24h) | SEP-EC rows | post-event (24h) ]

SEP-EC rows are carried through as VERBATIM TEXT except for the 17 CME-attribute
columns, which are replaced with the values recomputed over the full widened window
(Option B — Philip's rule 1d applied uniformly, so CMEs that began before the old
12h boundary are no longer invisible). Everything he models against — all flux/lag/
max columns, the target, delta_log_Intensity, Sunspot Number, the 4 CME-history
counters, timestamps — stays byte-identical.

Columns are renamed per Philip 1c:  Proton Intensity -> p16.4_tplus6,  p_* -> p16.4_*

Restored rows are trimmed by, in order:
  pre  : previous catalog event's end, then >=80% cumulative raw coverage (walking back)
  post : next catalog onset, rebound guard (only while well above background),
         then >=80% cumulative raw coverage
"""
import glob, os, re
import numpy as np
import pandas as pd

ROOT = "/Users/josiasmoukpe/Desktop/florida tech exit/sep-forecasting-research"
SHIP = os.path.join(ROOT, 'full')
EXT = os.path.join(ROOT, 'rebuild24')
OUT = os.path.join(ROOT, 'SEP-EC-24h-exact')
FULL = os.path.join(OUT, 'full')
EXTRA = os.path.join(OUT, 'extra_catalog_events')

MIN_BACKED = 0.8
REBOUND_MAX = 3.0
REBOUND_FLOOR_MULT = 10.0     # only police rebounds while >10x the event background
TSCOLS = ['Timestamp', 'Target Timestamp']

CME_ATTR_COLS = ['cme_donki_time', 'CME_DONKI_latitude', 'CME_DONKI_longitude',
                 'CME_DONKI_speed', 'CME_CDAW_MPA', 'CME_CDAW_LinearSpeed', 'VlogV',
                 'DONKI_half_width', 'Accelaration', '2nd_order_speed_final',
                 '2nd_order_speed_20R', 'CPA', 'Halo', 'Type2_Viz_Area',
                 'solar_wind_speed', 'diffusive_shock', 'half_richardson_value']


def ren(c):
    if c == 'Proton Intensity':
        return 'p16.4_tplus6'
    m = re.match(r'^p_(tminus\d+|t|max_intensity)$', c)
    return f'p16.4_{m.group(1)}' if m else c


def fmt(ts):
    return f"{ts.month}/{ts.day}/{ts.year} {ts.hour}:{ts.minute:02d}"


def fmt_cme(x):
    sx = str(x).strip()
    if sx in ('', '0', '0.0', 'nan', 'NaT'):
        return '0'
    return fmt(pd.Timestamp(sx))


def keep_prefix_by_coverage(block, backwards=False):
    """Longest prefix (or suffix) whose cumulative raw coverage stays >= MIN_BACKED."""
    if not len(block):
        return block
    b = block['flux_raw_backed'].values.astype(int)
    if backwards:
        b = b[::-1]
    frac = np.cumsum(b) / np.arange(1, len(b) + 1)
    good = np.where(frac >= MIN_BACKED)[0]
    if not len(good):
        return block.iloc[:0]
    n = good[-1] + 1
    return block.iloc[-n:] if backwards else block.iloc[:n]


def main():
    for d in (FULL, EXTRA):
        os.makedirs(d, exist_ok=True)
        for f in glob.glob(os.path.join(d, '*.csv')):
            os.remove(f)

    cat = pd.read_csv(os.path.join(ROOT, 'curr_pf10th10_original.csv'))
    cat['datetime'] = pd.to_datetime(cat['datetime'])
    onsets = np.sort(cat[cat['Index'] == 1]['datetime'].values)
    ends = np.sort(cat[cat['Index'] == 4]['datetime'].values)

    ship_txt, ship_key = {}, {}
    for f in glob.glob(os.path.join(SHIP, '*.csv')):
        ev = int(re.search(r'event_(\d+)_', os.path.basename(f)).group(1))
        t = pd.read_csv(f, dtype=str, keep_default_na=False)
        ship_txt[ev] = t
        ship_key[ev] = pd.to_datetime(t['Target Timestamp'])

    ext = {}
    for f in glob.glob(os.path.join(EXT, '*_ext24.csv')):
        ev = int(re.search(r'event_(\d+)_', os.path.basename(f)).group(1))
        d = pd.read_csv(f)
        for c in ('Timestamp', 'Target Timestamp', 'onset', 'catalog_end'):
            d[c] = pd.to_datetime(d[c])
        ext[ev] = d

    pair, used = {}, set()
    for ev in sorted(ship_txt):
        e = ship_key[ev].iloc[-1]
        c = sorted(((abs((ext[k]['catalog_end'].iloc[0] - e).total_seconds()), k)
                    for k in ext if k not in used))
        assert c and c[0][0] <= 600, f"no match for shipped #{ev}"
        pair[ev] = c[0][1]
        used.add(c[0][1])

    schema = [ren(c) for c in next(iter(ship_txt.values())).columns]
    rows, changelog = [], []

    for ev in sorted(ship_txt):
        s_raw, r = ship_txt[ev], ext[pair[ev]]
        first, last = ship_key[ev].iloc[0], ship_key[ev].iloc[-1]
        onset, cend = r['onset'].iloc[0], r['catalog_end'].iloc[0]
        assert set(ship_key[ev]) <= set(r['Target Timestamp']), f"coverage gap ev{ev}"

        # ---- pre block: everything before his first row, no cuts ----
        pre = r[r['Target Timestamp'] < first].copy()
        prev = ends[ends < np.datetime64(onset)]
        prev_end = pd.Timestamp(prev[-1]) if len(prev) else None

        # ---- post block: everything after his last row, no cuts ----
        post = r[r['Target Timestamp'] > last].copy()
        nxt = onsets[onsets > np.datetime64(last)]
        next_onset = pd.Timestamp(nxt[0]) if len(nxt) else None

        # ---- SEP-EC rows: verbatim text, CME attribute columns swapped in ----
        sv = pd.DataFrame({ren(c): s_raw[c] for c in s_raw.columns})
        sv['is_reconstructed'] = '0'
        changelog.append(dict(event=ev, cme_cells_changed=0, cme_rows_changed=0,
                              shipped_rows=len(sv)))

        def block(src, flag):
            h = pd.DataFrame(index=range(len(src)))
            for c in s_raw.columns:
                nc = ren(c)
                if c in TSCOLS:
                    h[nc] = [fmt(t) for t in src[c]]
                elif c in CME_ATTR_COLS:
                    # CME features are OFF outside the event (Philip 1d, clarified)
                    h[nc] = '0'
                elif c == 'Event ID':
                    h[nc] = s_raw['Event ID'].iloc[0]
                else:
                    h[nc] = src[nc].values
            h = h.astype(str)
            h['is_reconstructed'] = flag
            return h

        parts = [b for b in (block(pre, '1') if len(pre) else None, sv,
                             block(post, '2') if len(post) else None) if b is not None]
        out = pd.concat(parts, ignore_index=True)[schema + ['is_reconstructed']]
        out.to_csv(os.path.join(FULL, f'sep_event_{ev}_24h.csv'), index=False)

        k = pd.to_datetime(out['Target Timestamp'])
        dif = k.diff().dropna().unique()
        rows.append(dict(
            event=ev, rows_total=len(out), rows_shipped=len(s_raw),
            rows_pre=len(pre), rows_post=len(post),
            hours_pre=round(len(pre) * 5 / 60, 2), hours_post=round(len(post) * 5 / 60, 2),
            pre_raw_backed_pct=round(100 * float(pre['flux_raw_backed'].mean()), 1) if len(pre) else None,
            post_raw_backed_pct=round(100 * float(post['flux_raw_backed'].mean()), 1) if len(post) else None,
            onset=onset, catalog_end=cend, start=k.iloc[0], end=k.iloc[-1],
            prev_event_end=prev_end, next_onset=next_onset,
            contiguous_5min=(len(dif) == 1 and dif[0] == np.timedelta64(5, 'm')),
            overlaps_prev_event=bool(prev_end is not None and len(pre)
                                     and pre['Target Timestamp'].iloc[0] <= prev_end),
            overlaps_next_event=bool(next_onset is not None and len(post)
                                     and post['Target Timestamp'].iloc[-1] >= next_onset),
            era='2012' if first.year == 2012 else 'other'))

    # the two catalog events SEP-EC dropped, same schema
    for rv in sorted(set(ext) - used):
        src = ext[rv]
        inside = ((src['Target Timestamp'] >= src['onset'].iloc[0]) &
                  (src['Target Timestamp'] <= src['catalog_end'].iloc[0])).values
        d = pd.DataFrame(index=range(len(src)))
        for c in next(iter(ship_txt.values())).columns:
            nc = ren(c)
            if c in TSCOLS:
                d[nc] = [fmt(t) for t in src[c]]
            elif c == 'cme_donki_time':
                d[nc] = np.where(inside, [fmt_cme(x) for x in src[c]], '0')
            elif c in CME_ATTR_COLS:
                d[nc] = np.where(inside, src[nc].astype(str).values, '0')
            elif c == 'Event ID':
                d[nc] = rv
            else:
                d[nc] = src[nc].values
        d = d.astype(str)
        d['is_reconstructed'] = '1'
        d.to_csv(os.path.join(EXTRA, f'catalog_event_{rv}_24h.csv'), index=False)

    m = pd.DataFrame(rows)
    m.to_csv(os.path.join(OUT, 'manifest.csv'), index=False)
    pd.DataFrame(changelog).to_csv(os.path.join(OUT, 'cme_changelog.csv'), index=False)
    print(f"events={len(m)}  rows={m.rows_total.sum():,} = shipped {m.rows_shipped.sum():,} "
          f"+ pre {m.rows_pre.sum():,} + post {m.rows_post.sum():,}")
    print(f"pre  : {m.hours_pre.sum():.1f} h across {(m.rows_pre>0).sum()} events "
          f"(median {m.hours_pre.median():.1f} h)")
    print(f"post : {m.hours_post.sum():.1f} h across {(m.rows_post>0).sum()} events "
          f"(median {m.hours_post.median():.1f} h)")
    print(f"contiguous 5-min: {m.contiguous_5min.sum()}/{len(m)}")
    print(f"events with no pre : {sorted(m[m.rows_pre==0].event.tolist())}")
    print(f"events with no post: {sorted(m[m.rows_post==0].event.tolist())}")
    cl = pd.DataFrame(changelog)
    print(f"\nCME recompute: {int(cl.cme_cells_changed.sum()):,} cells changed across "
          f"{int((cl.cme_cells_changed>0).sum())} events")


if __name__ == '__main__':
    main()
