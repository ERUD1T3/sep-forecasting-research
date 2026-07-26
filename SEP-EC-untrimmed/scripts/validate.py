"""
Correct validation: the shipped (trimmed) file is a row-SUFFIX of the untrimmed
series, so every shipped row must appear in the regenerated UNTRIMMED file at the
same Target Timestamp. Match events by end-timestamp, join on Target Timestamp,
compare every column on the overlap. Trim-start behaviour is reported separately.
"""
import glob, os, re
import numpy as np
import pandas as pd

ROOT = "/Users/josiasmoukpe/Desktop/florida tech exit/sep-forecasting-research"

def load(p):
    d = pd.read_csv(p)
    d['Target Timestamp'] = pd.to_datetime(d['Target Timestamp'])
    return d

ship = {int(re.search(r'event_(\d+)_', os.path.basename(f)).group(1)): load(f)
        for f in glob.glob(os.path.join(ROOT, 'full', '*.csv'))}
unt = {int(re.search(r'event_(\d+)_', os.path.basename(f)).group(1)): load(f)
       for f in glob.glob(os.path.join(ROOT, 'rebuild', '*_filled_ie.csv'))}
trim = {int(re.search(r'event_(\d+)_', os.path.basename(f)).group(1)): load(f)
        for f in glob.glob(os.path.join(ROOT, 'rebuild', '*_ie_trim.csv'))}

# ---- match by END timestamp (catalog-determined, independent of flux) ----
unt_end = {k: v['Target Timestamp'].iloc[-1] for k, v in unt.items()}
pairs, unmatched = [], []
used = set()
for ev in sorted(ship):
    e = ship[ev]['Target Timestamp'].iloc[-1]
    cands = sorted(((abs((unt_end[k] - e).total_seconds()), k) for k in unt_end if k not in used))
    if cands and cands[0][0] <= 600:
        pairs.append((ev, cands[0][1], cands[0][0]))
        used.add(cands[0][1])
    else:
        unmatched.append((ev, cands[0] if cands else None))

print(f"shipped={len(ship)}  regenerated={len(unt)}")
print(f"TIER 1  matched by end-time (<=10min): {len(pairs)}/{len(ship)}")
for ev, c in unmatched:
    print(f"   UNMATCHED ship#{ev} best={c}")
print(f"  regenerated events with no shipped counterpart: {sorted(set(unt) - used)}\n")

pref = ['e0.5', 'e1.8', 'e4.4', 'p6.1', 'p33.0', 'p']
FLUX = ['Proton Intensity'] + [f'{p}_tminus{i}' for p in pref for i in range(24, 0, -1)] + \
       [f'{p}_t' for p in pref] + [f'{p}_max_intensity' for p in pref] + ['delta_log_Intensity']
EXACT = ['Sunspot Number', 'CME_DONKI_latitude', 'CME_DONKI_longitude', 'CME_DONKI_speed',
         'CME_CDAW_MPA', 'CME_CDAW_LinearSpeed', 'VlogV', 'DONKI_half_width', 'Accelaration',
         '2nd_order_speed_final', '2nd_order_speed_20R', 'CPA', 'Halo', 'Type2_Viz_Area',
         'solar_wind_speed', 'diffusive_shock', 'half_richardson_value', 'CMEs Past Month',
         'CMEs Past 9 Hours', 'CMEs Speed > 1000', 'Max CME Speed']

col_stats = {}
cover_fail, trim_exact, trim_rows = [], 0, []
rows_cmp = 0
per_event = []
for sev, rev, cost in pairs:
    a, b = ship[sev], unt[rev]
    # coverage: every shipped timestamp present in untrimmed?
    sa = set(a['Target Timestamp']); sb = set(b['Target Timestamp'])
    missing = len(sa - sb)
    if missing:
        cover_fail.append((sev, rev, missing, len(sa)))
    # trim reproduction
    t = trim[rev]
    same_trim = len(t) == len(a) and (t['Target Timestamp'].values == a['Target Timestamp'].values).all()
    trim_exact += int(same_trim)
    trim_rows.append((sev, rev, len(a), len(t), len(b)))
    # join on Target Timestamp
    m = a.merge(b, on='Target Timestamp', suffixes=('_s', '_r'))
    rows_cmp += len(m)
    ev_ok = ev_tot = 0
    for c in EXACT + FLUX:
        cs, cr = f'{c}_s', f'{c}_r'
        if cs not in m or cr not in m:
            continue
        x = pd.to_numeric(m[cs], errors='coerce'); y = pd.to_numeric(m[cr], errors='coerce')
        ok = np.isclose(x, y, rtol=1e-6, atol=1e-9, equal_nan=True) | (x.isna() & y.isna())
        d = col_stats.setdefault(c, [0, 0, 0.0])
        d[0] += int(ok.sum()); d[1] += len(ok)
        ev_ok += int(ok.sum()); ev_tot += len(ok)
        if (~ok).any():
            dd = np.abs(x - y)[~ok]
            if len(dd) and np.isfinite(dd).any():
                d[2] = max(d[2], float(np.nanmax(dd)))
    per_event.append((sev, rev, 100.0 * ev_ok / ev_tot if ev_tot else 0, len(m), len(a)))

print(f"TIER 2  coverage (every shipped row present in regenerated untrimmed):")
print(f"  events fully covered: {len(pairs) - len(cover_fail)}/{len(pairs)}")
for sev, rev, miss, tot in cover_fail[:10]:
    print(f"    ship#{sev}/reb#{rev}: {miss}/{tot} shipped timestamps absent")
print(f"  rows compared: {rows_cmp:,}\n")

print(f"TIER 3  trim reproduction: exact on {trim_exact}/{len(pairs)} events")
off = [(s, r, ns, nt) for s, r, ns, nt, nu in trim_rows if ns != nt]
print(f"  events where trim length differs: {len(off)}")
for s, r, ns, nt in off[:10]:
    print(f"    ship#{s}/reb#{r}: shipped {ns} rows, my trim {nt} rows (diff {nt-ns:+d})")
print()

def report(title, cols):
    print(title)
    rows = []
    for c in cols:
        if c in col_stats:
            ok, tot, mx = col_stats[c]
            rows.append((100.0 * ok / tot, c, mx, tot - ok, tot))
    rows.sort()
    perfect = sum(1 for r in rows if r[0] == 100.0)
    print(f"  columns 100% matching: {perfect}/{len(rows)}")
    for pct, c, mx, nd, tot in rows[:8]:
        if pct < 100.0:
            print(f"    {c:<26} {pct:7.3f}%  max|diff|={mx:.6g}  ({nd}/{tot} differ)")
    print()

report("TIER 4  exact-source columns (CME + sunspot):", EXACT)
report("TIER 5  flux-derived columns:", FLUX)

tot_ok = sum(v[0] for v in col_stats.values()); tot_all = sum(v[1] for v in col_stats.values())
print(f"OVERALL cell agreement on overlap: {tot_ok:,}/{tot_all:,} = {100.0*tot_ok/tot_all:.4f}%\n")
per_event.sort(key=lambda x: x[2])
print("worst events by agreement:")
for sev, rev, pct, nm, na in per_event[:8]:
    print(f"   ship#{sev:>2} / reb#{rev:>2}: {pct:7.3f}%  joined {nm}/{na} rows")
print("\nbest events:")
for sev, rev, pct, nm, na in per_event[-3:]:
    print(f"   ship#{sev:>2} / reb#{rev:>2}: {pct:7.3f}%  joined {nm}/{na} rows")
