"""
Improved 2012 flux harvest.

Each absolute timestamp T is observed MANY times across the derived files:
as `{ch}_t` in one row, `{ch}_tminus1` in the next, ... `{ch}_tminus24` 24 rows
later, and in every event whose window covers T. Because interpolate_and_extrapolate
runs per COLUMN, timestamps that were NaN in the raw flux get a DIFFERENT filled
value in each column. The genuine raw value is therefore the MODAL observation,
while interpolated artefacts are scattered singletons.

Strategy: collect every observation of (channel, timestamp), then take the value
with the highest multiplicity (ties -> median of the tied values).
"""
import glob, io, os, re, subprocess, pickle
from collections import defaultdict, Counter
import numpy as np
import pandas as pd

ROOT = "/Users/josiasmoukpe/Desktop/florida tech exit/sep-forecasting-research"
OUTPKL = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'harvest2012_mode.pkl')

CH_FULL = {'e0.5': 'e0.5', 'e1.8': 'e1.8', 'e4.4': 'e4.4',
           'p6.10': 'p6.1', 'p33.00': 'p33.0', 'p16.40': 'p'}
CH_V5 = {'e0.5': 'e0.5', 'e1.8': 'e1.8', 'e4.4': 'e4.4', 'p16.40': 'p'}

obs = {c: defaultdict(list) for c in CH_FULL}


def ingest(df, chans, year=2012):
    ts = pd.to_datetime(df['Timestamp'])
    tt = pd.to_datetime(df['Target Timestamp'])
    for c, src in chans.items():
        for k in range(24, 0, -1):
            col = f'{src}_tminus{k}'
            if col not in df.columns:
                continue
            key = ts - pd.Timedelta(minutes=5 * k)
            v = df[col].values
            for t, val in zip(key, v):
                if t.year == year and val != -9999 and np.isfinite(val):
                    obs[c][t].append(val)
        col = f'{src}_t'
        if col in df.columns:
            for t, val in zip(ts, df[col].values):
                if t.year == year and val != -9999 and np.isfinite(val):
                    obs[c][t].append(val)
    if 'p16.40' in chans:
        for t, val in zip(tt, df['Proton Intensity'].values):
            if t.year == year and val != -9999 and np.isfinite(val):
                obs['p16.40'][t].append(val)


print("ingesting shipped misc/full ...", flush=True)
for f in glob.glob(os.path.join(ROOT, 'full', '*.csv')):
    d = pd.read_csv(f)
    t0 = pd.to_datetime(d['Timestamp'])
    if t0.iloc[0].year != 2012 and t0.iloc[-1].year != 2012:
        continue
    ingest(d, CH_FULL)

print("ingesting git snapshots ...", flush=True)
# use several snapshots; duplicates only strengthen the mode
for commit in ['755a4e2', 'a95564e', 'a0a43c0', 'cb2d04f']:
    files = subprocess.run(['git', 'ls-tree', '-r', commit, '--name-only'],
                           capture_output=True, text=True, cwd=ROOT).stdout.splitlines()
    ev = [p for p in files if re.search(r'sep_event_\d+_filled_ie\.csv$', p)]
    seen = set()
    n = 0
    for p in ev:
        bn = os.path.basename(p)
        if bn in seen:
            continue
        seen.add(bn)
        blob = subprocess.run(['git', 'show', f'{commit}:{p}'],
                              capture_output=True, text=True, cwd=ROOT).stdout
        try:
            d = pd.read_csv(io.StringIO(blob))
        except Exception:
            continue
        t0 = pd.to_datetime(d['Timestamp'])
        if t0.iloc[0].year != 2012 and t0.iloc[-1].year != 2012:
            continue
        ingest(d, CH_V5)
        n += 1
    print(f"   {commit}: ingested {n} 2012 event files", flush=True)


def consolidate(vals):
    """Modal value; ties broken by median of the most common values."""
    if len(vals) == 1:
        return vals[0]
    cnt = Counter(np.round(vals, 12))
    top = max(cnt.values())
    best = [v for v, c in cnt.items() if c == top]
    return float(np.median(best))


final = {}
print("\nconsolidating:")
for c in obs:
    d = {}
    multi = uniq = 0
    for t, vals in obs[c].items():
        if len(vals) > 1:
            multi += 1
            if len(set(np.round(vals, 12))) > 1:
                uniq += 1
        d[t] = consolidate(vals)
    final[c] = d
    print(f"   {c:>7}: {len(d):,} timestamps | {multi:,} multi-observed | "
          f"{uniq:,} inconsistent (interpolation artefacts resolved)")

pickle.dump(final, open(OUTPKL, 'wb'))
print(f"\nsaved -> {OUTPKL}")
