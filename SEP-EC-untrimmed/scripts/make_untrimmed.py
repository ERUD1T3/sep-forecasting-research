"""
Build the definitive UNTRIMMED SEP-EC dataset (byte-faithful edition).

Shipped rows are carried through as VERBATIM TEXT (every column read as str), so
the output is byte-identical to misc/full on every row SEP-EC already had - no
float re-formatting, no timestamp re-formatting. Only the restored pre-onset
background rows are reconstructed, formatted to match the native
`M/D/YYYY H:MM` timestamp style, and flagged with is_reconstructed=1.
"""
import glob, os, re
import numpy as np
import pandas as pd

ROOT = "/Users/josiasmoukpe/Desktop/florida tech exit/sep-forecasting-research"
SHIP = os.path.join(ROOT, 'full')
OUT = os.path.join(ROOT, 'SEP-EC-untrimmed')
FULL = os.path.join(OUT, 'full')
EXTRA = os.path.join(OUT, 'extra_catalog_events')

TSCOLS = ['Timestamp', 'Target Timestamp']


def fmt(ts):
    """Native SEP-EC timestamp style: M/D/YYYY H:MM, unpadded."""
    return f"{ts.month}/{ts.day}/{ts.year} {ts.hour}:{ts.minute:02d}"


def main():
    for d in (FULL, EXTRA):
        os.makedirs(d, exist_ok=True)

    # shipped: raw text, plus parsed key for alignment
    ship_txt, ship_key = {}, {}
    for f in glob.glob(os.path.join(SHIP, '*.csv')):
        ev = int(re.search(r'event_(\d+)_', os.path.basename(f)).group(1))
        t = pd.read_csv(f, dtype=str, keep_default_na=False)
        ship_txt[ev] = t
        ship_key[ev] = pd.to_datetime(t['Target Timestamp'])

    rec = {}
    for sub in ('rebuild', 'rebuild_patched'):
        for f in glob.glob(os.path.join(ROOT, sub, '*_filled_ie.csv')):
            rec[int(re.search(r'event_(\d+)_', os.path.basename(f)).group(1))] = f
    recd = {}
    for k, v in rec.items():
        d = pd.read_csv(v)
        d['Target Timestamp'] = pd.to_datetime(d['Target Timestamp'])
        d['Timestamp'] = pd.to_datetime(d['Timestamp'])
        recd[k] = d

    rend = {k: v['Target Timestamp'].iloc[-1] for k, v in recd.items()}
    pair, used = {}, set()
    for ev in sorted(ship_txt):
        e = ship_key[ev].iloc[-1]
        c = sorted(((abs((rend[k] - e).total_seconds()), k) for k in rend if k not in used))
        assert c and c[0][0] <= 600, f"no match for #{ev}"
        pair[ev] = c[0][1]
        used.add(c[0][1])

    rows = []
    for ev in sorted(ship_txt):
        s, r = ship_txt[ev], recd[pair[ev]]
        cols = list(s.columns)
        first = ship_key[ev].iloc[0]
        head = r[r['Target Timestamp'] < first].copy()
        assert set(ship_key[ev]) <= set(r['Target Timestamp']), f"coverage gap ev{ev}"

        if len(head):
            h = pd.DataFrame(index=range(len(head)))
            for c in cols:
                if c in TSCOLS:
                    h[c] = [fmt(t) for t in head[c]]
                elif c == 'Event ID':
                    h[c] = s['Event ID'].iloc[0]
                else:
                    h[c] = head[c].values
            h = h.astype(str)
            h['is_reconstructed'] = '1'
            sv = s.copy()
            sv['is_reconstructed'] = '0'
            out = pd.concat([h, sv], ignore_index=True)
        else:
            out = s.copy()
            out['is_reconstructed'] = '0'

        out.to_csv(os.path.join(FULL, f'sep_event_{ev}_filled_ie.csv'), index=False)
        k = pd.to_datetime(out['Target Timestamp'])
        d = k.diff().dropna().unique()
        rows.append(dict(event=ev, rows_total=len(out), rows_shipped=len(s),
                         rows_restored=len(head),
                         hours_restored=round(len(head) * 5 / 60, 2),
                         start=k.iloc[0], shipped_start=first, end=k.iloc[-1],
                         contiguous_5min=(len(d) == 1 and d[0] == np.timedelta64(5, 'm')),
                         restored_source=('patched-2012' if first.year == 2012 else 'raw-v1')
                         if len(head) else 'none'))

    for rv in sorted(set(recd) - used):
        d = recd[rv].copy()
        for c in TSCOLS:
            d[c] = [fmt(t) for t in d[c]]
        d['is_reconstructed'] = 1
        d.to_csv(os.path.join(EXTRA, f'catalog_event_{rv}_filled_ie.csv'), index=False)

    man = pd.DataFrame(rows)
    man.to_csv(os.path.join(OUT, 'manifest.csv'), index=False)
    print(f"events={len(man)}  rows={man.rows_total.sum():,} "
          f"(shipped {man.rows_shipped.sum():,} + restored {man.rows_restored.sum():,})")
    print(f"restored background: {man.hours_restored.sum():.1f} h across "
          f"{(man.rows_restored > 0).sum()} events")
    print(f"contiguous 5-min: {man.contiguous_5min.sum()}/{len(man)}")


if __name__ == '__main__':
    main()
