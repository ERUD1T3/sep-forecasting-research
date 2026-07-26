"""
Final untrimmed + extended SEP-EC build.

Per event:
  [ restored pre-onset background ] + [ SEP-EC rows, VERBATIM ] + [ restored post-event background ]

`is_reconstructed`: 0 = original SEP-EC row (byte-identical), 1 = restored before
onset, 2 = restored after the catalog event end.

Post-event rows are only emitted while the flux file genuinely had data: rows are
included up to the last index where the cumulative raw-backed fraction stays
>= MIN_BACKED (0.8), so a sparse region cannot spawn a long interpolated tail.
Events with no usable post-event flux (the 2012 blackout, plus event 7 whose
post window is only ~44% covered in the raw file) simply get no post rows.
"""
import glob, os, re
import numpy as np
import pandas as pd

ROOT = "/Users/josiasmoukpe/Desktop/florida tech exit/sep-forecasting-research"
SHIP = os.path.join(ROOT, 'full')
EXT = os.path.join(ROOT, 'rebuild_extended')
OUT = os.path.join(ROOT, 'SEP-EC-untrimmed')
FULL = os.path.join(OUT, 'full')
EXTRA = os.path.join(OUT, 'extra_catalog_events')
TSCOLS = ['Timestamp', 'Target Timestamp']
MIN_BACKED = 0.8


def fmt(ts):
    return f"{ts.month}/{ts.day}/{ts.year} {ts.hour}:{ts.minute:02d}"


def usable_post(post):
    """Longest prefix whose cumulative raw-backed fraction stays >= MIN_BACKED."""
    if not len(post):
        return post.iloc[:0]
    b = post['flux_raw_backed'].values.astype(int)
    cum = np.cumsum(b)
    frac = cum / np.arange(1, len(b) + 1)
    good = np.where(frac >= MIN_BACKED)[0]
    if not len(good):
        return post.iloc[:0]
    return post.iloc[:good[-1] + 1]


def main():
    for d in (FULL, EXTRA):
        os.makedirs(d, exist_ok=True)
        for f in glob.glob(os.path.join(d, '*.csv')):
            os.remove(f)

    ship_txt, ship_key = {}, {}
    for f in glob.glob(os.path.join(SHIP, '*.csv')):
        ev = int(re.search(r'event_(\d+)_', os.path.basename(f)).group(1))
        t = pd.read_csv(f, dtype=str, keep_default_na=False)
        ship_txt[ev] = t
        ship_key[ev] = pd.to_datetime(t['Target Timestamp'])

    ext = {}
    for f in glob.glob(os.path.join(EXT, '*_ext.csv')):
        ev = int(re.search(r'event_(\d+)_', os.path.basename(f)).group(1))
        d = pd.read_csv(f)
        d['Target Timestamp'] = pd.to_datetime(d['Target Timestamp'])
        d['Timestamp'] = pd.to_datetime(d['Timestamp'])
        d['catalog_end'] = pd.to_datetime(d['catalog_end'])
        ext[ev] = d

    # match shipped -> extended by the catalog end time
    pair, used = {}, set()
    for ev in sorted(ship_txt):
        e = ship_key[ev].iloc[-1]
        c = sorted(((abs((ext[k]['catalog_end'].iloc[0] - e).total_seconds()), k)
                    for k in ext if k not in used))
        assert c and c[0][0] <= 600, f"no extended match for shipped #{ev} (best {c[0][0]}s)"
        pair[ev] = c[0][1]
        used.add(c[0][1])

    rows = []
    for ev in sorted(ship_txt):
        s, r = ship_txt[ev], ext[pair[ev]]
        cols = list(s.columns)
        first, last = ship_key[ev].iloc[0], ship_key[ev].iloc[-1]
        assert set(ship_key[ev]) <= set(r['Target Timestamp']), f"coverage gap ev{ev}"

        pre = r[r['Target Timestamp'] < first]
        post = usable_post(r[r['Target Timestamp'] > last])

        def block(src, flag):
            h = pd.DataFrame(index=range(len(src)))
            for c in cols:
                if c in TSCOLS:
                    h[c] = [fmt(t) for t in src[c]]
                elif c == 'Event ID':
                    h[c] = s['Event ID'].iloc[0]
                else:
                    h[c] = src[c].values
            h = h.astype(str)
            h['is_reconstructed'] = flag
            return h

        sv = s.copy()
        sv['is_reconstructed'] = '0'
        parts = [b for b in (block(pre, '1') if len(pre) else None, sv,
                             block(post, '2') if len(post) else None) if b is not None]
        out = pd.concat(parts, ignore_index=True)
        out.to_csv(os.path.join(FULL, f'sep_event_{ev}_filled_ie.csv'), index=False)

        k = pd.to_datetime(out['Target Timestamp'])
        diffs = k.diff().dropna().unique()
        pi = pd.to_numeric(out['Proton Intensity'])
        imax = int(pi.idxmax())
        rows.append(dict(
            event=ev, rows_total=len(out), rows_shipped=len(s),
            rows_pre=len(pre), rows_post=len(post),
            post_raw_backed_pct=(round(100 * float(post['flux_raw_backed'].mean()), 1)
                                 if len(post) else None),
            hours_pre=round(len(pre) * 5 / 60, 2), hours_post=round(len(post) * 5 / 60, 2),
            start=k.iloc[0], shipped_start=first, catalog_end=last, end=k.iloc[-1],
            contiguous_5min=(len(diffs) == 1 and diffs[0] == np.timedelta64(5, 'm')),
            peak_pos_pct=round(100 * imax / (len(out) - 1), 1),
            hours_after_peak=round((k.iloc[-1] - k.iloc[imax]).total_seconds() / 3600, 1),
            decades_decay=round(float(np.log10(max(pi.max(), 1e-12))
                                      - np.log10(max(pi.iloc[-1], 1e-12))), 2),
            era=('2012-blackout' if first.year == 2012 else 'raw-v1')))

    for rv in sorted(set(ext) - used):
        d = ext[rv].copy()
        for c in TSCOLS:
            d[c] = [fmt(t) for t in d[c]]
        d['is_reconstructed'] = 1
        d.to_csv(os.path.join(EXTRA, f'catalog_event_{rv}_filled_ie.csv'), index=False)

    m = pd.DataFrame(rows)
    m.to_csv(os.path.join(OUT, 'manifest.csv'), index=False)
    print(f"events={len(m)} rows={m.rows_total.sum():,} = shipped {m.rows_shipped.sum():,} "
          f"+ pre {m.rows_pre.sum():,} + post {m.rows_post.sum():,}")
    print(f"pre-onset restored : {m.hours_pre.sum():.1f} h across {(m.rows_pre>0).sum()} events")
    print(f"post-event restored: {m.hours_post.sum():.1f} h across {(m.rows_post>0).sum()} events")
    print(f"contiguous 5-min   : {m.contiguous_5min.sum()}/{len(m)}")
    print(f"events with no post extension: {sorted(m[m.rows_post==0].event.tolist())}")
    print(f"\ndecay after peak: median {m.decades_decay.median():.2f} decades over "
          f"{m.hours_after_peak.median():.1f} h")
    print(f"events with >=1 decade of decay: {(m.decades_decay>=1).sum()}/44 "
          f"(was 17/44 before extension)")


if __name__ == '__main__':
    main()
