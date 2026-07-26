"""
Regenerate every catalog event with the window EXTENDED PAST the event end, so the
decay can run out into post-event background.

The original pipeline stopped at the catalog end time (Index 4). Here the grid runs
  start = onset - HOURS_BEFORE   ...   end = catalog_end + HOURS_AFTER

A per-row mask `flux_raw_backed` records whether the flux file actually had a datum
at that timestamp BEFORE interpolation, so the splice step can truncate the tail
instead of shipping extrapolated filler (critical for 2012, where the harvest only
covers the original windows).
"""
import os, sys, pickle
from datetime import timedelta
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rebuild_untrimmed as R
from patch_rerun import apply_patch, CHANMAP  # noqa: F401  (patch uses harvest pickle)

HOURS_AFTER = 24
OUT = os.path.join(R.ROOT, 'rebuild_extended')


def build_time_grid_ext(group, hours_before=R.HOURS_BEFORE, hours_after=HOURS_AFTER):
    onset = group[group['Index'] == 1]['datetime'].iloc[0]
    peak = group[group['Index'] == 3]['datetime'].iloc[0]
    end = group[group['Index'] == 4]['datetime'].iloc[0]
    start = onset - timedelta(hours=hours_before)
    end_ext = end + timedelta(hours=hours_after)

    targets, cur = [], start
    while cur <= end_ext:
        targets.append(cur)
        cur += timedelta(minutes=5)
    if targets[-1] != end_ext:
        targets.append(cur)
    features = [t - timedelta(minutes=30) for t in targets]
    return onset, peak, start, end, end_ext, targets, features


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    print("loading flux ...", flush=True)
    flux = R.load_flux_data(R.FLUX_PATH)
    flux = apply_patch(flux)          # fill the 2012 blackout where harvest allows
    print("flux ready", flush=True)

    raw_ok = flux[R.PROTON_CHANNEL].notna()

    ds1 = pd.read_csv(R.SEP_PATH)
    ds1['datetime'] = pd.to_datetime(ds1['datetime'])
    ds1 = ds1[(ds1['Year'] >= 2010) & (ds1['Year'] <= 2017)]
    ds2 = pd.read_csv(R.CME_PATH)
    ds2['CME_DONKI_time'] = pd.to_datetime(ds2['CME_DONKI_time'], format='%m/%d/%Y %H:%M')
    R.ds2_daily = {}
    for _, r in ds2.iterrows():
        if pd.notna(r['CME_DONKI_time']):
            R.ds2_daily.setdefault(r['CME_DONKI_time'].date(), r['daily_sunspots'])
    valid = ds2.dropna(subset=['CME_DONKI_time']).sort_values('CME_DONKI_time')
    cme_sorted = (pd.DatetimeIndex(valid['CME_DONKI_time'].values),
                  valid['CME_DONKI_speed'].values.astype(float))
    sn = pd.read_csv(R.SN_PATH, delimiter=';', header=None,
                     names=['Year', 'Month', 'Day', 'Decimal Year', 'Sunspot Number',
                            'Standard Error', 'Observations', 'Definitive/Provisional'])
    sn['Date'] = pd.to_datetime(sn[['Year', 'Month', 'Day']])
    sn_dict = dict(zip(sn['Date'], sn['Sunspot Number']))

    # build_event() calls build_time_grid(group) expecting
    # (onset, peak, start, end, targets, features); swap in the extended grid.
    def _extended_grid(group):
        onset, peak, start, end, end_ext, targets, features = build_time_grid_ext(group)
        return onset, peak, start, end_ext, targets, features

    R.build_time_grid = _extended_grid

    groups = list(ds1.groupby(ds1.index // 4))
    for n, (_, g) in enumerate(groups, start=1):
        onset, peak, start, end, end_ext, targets, features = build_time_grid_ext(g)
        df, _, _ = R.build_event(n, g, ds2, flux, sn_dict, cme_sorted)
        # raw-backing mask BEFORE interpolation, evaluated at Target Timestamp
        idx = flux.index.get_indexer(pd.DatetimeIndex(df['Target Timestamp']), method='nearest')
        backed = raw_ok.values[idx]
        df = R.finalize(df, sn_dict)
        df['flux_raw_backed'] = backed.astype(int)
        df['catalog_end'] = end
        df.to_csv(os.path.join(OUT, f'sep_event_{n}_filled_ie_ext.csv'), index=False)
        post = df[df['Target Timestamp'] > end]
        nb = int(post['flux_raw_backed'].sum())
        print(f"  event {n:>2}: rows={len(df):>5} post-end={len(post):>4} "
              f"raw-backed={nb:>4} ({100*nb/max(len(post),1):5.1f}%)", flush=True)
    print("done ->", OUT)
