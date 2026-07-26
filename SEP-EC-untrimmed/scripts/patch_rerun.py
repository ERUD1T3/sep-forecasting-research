"""
Fill the 2012 flux blackout in ephin5m.dat (v1) by harvesting the flux values back
out of the derived datasets, then re-run the pipeline for the affected events.

Sources, in priority order:
  A) shipped misc/full  -- all 6 channels, covers the trimmed windows (+2h of lags)
  B) git v5 untrimmed   -- 4 channels (e0.5,e1.8,e4.4,p16.40), covers full windows
"""
import os, sys, pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rebuild_untrimmed as R

PATCH = pickle.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      'harvest2012.pkl'), 'rb'))
CHANMAP = {
    'e0.5': 'Electron_Flux_0.5MeV',
    'e1.8': 'Electron_Flux_1.8MeV',
    'e4.4': 'Electron_Flux_4.4MeV',
    'p6.10': 'Proton_Flux_6.10MeV',
    'p16.40': 'Proton_Flux_16.40MeV',
    'p33.00': 'Proton_Flux_33.00MeV',
}
OUT = os.path.join(R.ROOT, 'rebuild_patched')


def apply_patch(flux):
    merged = {}
    for c in CHANMAP:
        d = dict(PATCH['B'].get(c, {}))   # lower priority
        d.update(PATCH['A'].get(c, {}))   # shipped wins
        merged[c] = d
    # The generator read flux by NEAREST grid point, so invert by snapping each
    # harvested timestamp onto the nearest 5-minute grid row.
    for c, col in CHANMAP.items():
        d = merged.get(c)
        if not d:
            continue
        times = pd.DatetimeIndex(sorted(d))
        vals = np.array([d[t] for t in times], dtype=float)
        idx = flux.index.get_indexer(times, method='nearest')
        good = idx >= 0
        arr = flux[col].values.copy()
        arr[idx[good]] = vals[good]
        flux[col] = arr
        print(f"  patched {col:<24} {len(set(idx[good])):,} grid rows "
              f"(from {int(good.sum()):,} harvested)", flush=True)
    return flux


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    print("loading flux ...", flush=True)
    flux = R.load_flux_data(R.FLUX_PATH)
    before = flux.loc['2012', 'Proton_Flux_16.40MeV'].notna().sum()
    flux = apply_patch(flux)
    after = flux.loc['2012', 'Proton_Flux_16.40MeV'].notna().sum()
    print(f"2012 valid p16.40 rows: {before:,} -> {after:,}", flush=True)

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

    groups = list(ds1.groupby(ds1.index // 4))
    for n, (_, g) in enumerate(groups, start=1):
        onset = g[g['Index'] == 1]['datetime'].iloc[0]
        end = g[g['Index'] == 4]['datetime'].iloc[0]
        if not (onset.year == 2012 or end.year == 2012):
            continue
        df, _, _ = R.build_event(n, g, ds2, flux, sn_dict, cme_sorted)
        df = R.finalize(df, sn_dict)
        df.to_csv(os.path.join(OUT, f"sep_event_{n}_filled_ie.csv"), index=False)
        tr = R.trim_background(df)
        tr.to_csv(os.path.join(OUT, f"sep_event_{n}_filled_ie_trim.csv"), index=False)
        print(f"  event {n:>2}: untrimmed={len(df):>5} trim={len(tr):>5}", flush=True)
    print("done ->", OUT)
