"""
Reconstruct the SEP-EC (electron_cme) dataset from raw sources, producing the
UNTRIMMED per-event files, then verify by re-applying trim_background and
diffing against the shipped misc/full dataset.

Pipeline (mirrors notebooks/building_electron_ts_dataset.ipynb):
  1. window extraction   (process_group_data / generate_event_time_data)
  2. sunspot fill        (fill_missing_sunspot_numbers)
  3. interpolate/extrap  (interpolate_and_extrapolate)
  4. max intensity       (update_max_intensity_in_directory)
  5. delta_log_Intensity (preprocess.py)
  6. trim                (utils.trim_background)  -- for validation only

Two deviations from the committed notebook, established empirically:
  - hours_before = 12 (notebook says 16; 5 shipped events pin start to onset-12h)
  - channel prefixes p6.1 / p33.0 (notebook emits e6.10 / e33.00)
"""
import os
import sys
import numpy as np
import pandas as pd
from datetime import timedelta

ROOT = "/Users/josiasmoukpe/Desktop/florida tech exit/sep-forecasting-research"
OUT = os.path.join(ROOT, "rebuild")
SEP_PATH = os.path.join(ROOT, "curr_pf10th10_original.csv")
CME_PATH = os.path.join(ROOT, "SEP10MeV_Features_v2.csv")
FLUX_PATH = os.path.join(ROOT, "ephin5m.dat")
SN_PATH = os.path.join(ROOT, "SN_d_tot_V2.0.csv")

HOURS_BEFORE = 12
CME_OFFSET = timedelta(minutes=30)
AVG_SEP_DURATION = timedelta(days=0, hours=20, minutes=13, seconds=56, microseconds=86956)
PROTON_CHANNEL = "Proton_Flux_16.40MeV"
CHANNELS = [
    ("Electron_Flux_0.5MeV", "e0.5"),
    ("Electron_Flux_1.8MeV", "e1.8"),
    ("Electron_Flux_4.4MeV", "e4.4"),
    ("Proton_Flux_6.10MeV", "p6.1"),
    ("Proton_Flux_33.00MeV", "p33.0"),
]
CME_ATTRS = ['CME_DONKI_latitude', 'CME_DONKI_longitude', 'CME_DONKI_speed',
             'CME_CDAW_MPA', 'CME_CDAW_LinearSpeed', 'VlogV', 'DONKI_half_width',
             'Accelaration', '2nd_order_speed_final', '2nd_order_speed_20R',
             'CPA', 'Halo', 'Type2_Viz_Area', 'solar_wind_speed', 'diffusive_shock',
             'half_richardson_value']
CME_KEYS = ['cme_donki_time'] + CME_ATTRS


def load_flux_data(filepath):
    """Same semantics as modules/training/electron_ts.py:load_flux_data."""
    df = pd.read_csv(filepath, sep=r"\s+", header=None, skiprows=4)
    df.columns = ['Year', 'Seconds_of_year', 'Electron_Flux_0.5MeV', 'Electron_Flux_1.8MeV',
                  'Electron_Flux_4.4MeV', 'Electron_Flux_7.4MeV', 'Proton_Flux_0.54MeV',
                  'Proton_Flux_1.37MeV', 'Proton_Flux_4.01MeV', 'Proton_Flux_6.10MeV',
                  'Proton_Flux_16.40MeV', 'Proton_Flux_33.00MeV', 'Proton_Flux_47.00MeV']
    df['Time'] = pd.to_datetime(df['Year'].astype(int).astype(str), format='%Y') + \
        pd.to_timedelta(df['Seconds_of_year'].astype('int64'), unit='s')
    df = df.set_index('Time')
    df = df.replace(-9.9999998E+30, np.nan)
    return df


def nearest_lookup(flux_index, values, times):
    """Vectorised equivalent of ds3_flux.index.get_loc(t, method='nearest')."""
    idx = flux_index.get_indexer(pd.DatetimeIndex(times), method='nearest')
    return values[idx]


def build_time_grid(group):
    onset = group[group['Index'] == 1]['datetime'].iloc[0]
    peak = group[group['Index'] == 3]['datetime'].iloc[0]
    end = group[group['Index'] == 4]['datetime'].iloc[0]
    start = onset - timedelta(hours=HOURS_BEFORE)

    targets, cur = [], start
    while cur <= end:
        targets.append(cur)
        cur += timedelta(minutes=5)
    if targets[-1] != end:
        targets.append(cur)
    features = [t - timedelta(minutes=30) for t in targets]
    return onset, peak, start, end, targets, features


def cme_stats_vectorised(cme_times_sorted, cme_speeds_sorted, features):
    """Vectorised cme_statistics_for_row: current > t >= current - H, speed > 1000."""
    f = pd.DatetimeIndex(features)
    T = cme_times_sorted

    def window(hours):
        hi = T.searchsorted(f, side='left')
        lo = T.searchsorted(f - pd.Timedelta(hours=hours), side='left')
        return lo, hi

    lo_m, hi_m = window(30 * 24)
    month = hi_m - lo_m
    lo_9, hi_9 = window(9)
    nine = hi_9 - lo_9

    fast = np.cumsum(np.concatenate([[0], (cme_speeds_sorted > 1000).astype(int)]))
    nine_fast = fast[hi_9] - fast[lo_9]

    lo_d, hi_d = window(24)
    maxspeed = np.zeros(len(f))
    for i in range(len(f)):
        if hi_d[i] > lo_d[i]:
            seg = cme_speeds_sorted[lo_d[i]:hi_d[i]]
            maxspeed[i] = np.nanmax(seg) if np.isfinite(seg).any() else 0
    return month, nine, nine_fast, maxspeed


def build_event(event_id, group, ds2, flux, sn_dict, cme_sorted):
    onset, peak, start, end, targets, features = build_time_grid(group)
    fidx = flux.index

    data = {
        'Event ID': [event_id] * len(features),
        'Timestamp': features,
        'Target Timestamp': targets,
        'Proton Intensity': nearest_lookup(fidx, flux[PROTON_CHANNEL].values, targets),
    }

    for channel, prefix in CHANNELS:
        vals = flux[channel].values
        for i in range(24, 0, -1):
            shifted = [t - timedelta(minutes=i * 5) for t in features]
            data[f'{prefix}_tminus{i}'] = nearest_lookup(fidx, vals, shifted)
        data[f'{prefix}_t'] = nearest_lookup(fidx, vals, features)
        data[f'{prefix}_max_intensity'] = np.zeros(len(features))  # recomputed post-ie

    pvals = flux[PROTON_CHANNEL].values
    for i in range(24, 0, -1):
        shifted = [t - timedelta(minutes=i * 5) for t in features]
        data[f'p_tminus{i}'] = nearest_lookup(fidx, pvals, shifted)
    data['p_t'] = nearest_lookup(fidx, pvals, features)
    data['p_max_intensity'] = np.zeros(len(features))

    # Sunspot number: match ds2 CME date -> daily_sunspots, else NaN (filled later)
    sun = []
    for t in features:
        v = ds2_daily.get(t.date())
        sun.append(v if v is not None else np.nan)
    data['Sunspot Number'] = sun

    # CME attributes: forward-fill from each CME onset for AVG_SEP_DURATION
    cmes = {t: 0 for t in features}
    span = AVG_SEP_DURATION // timedelta(minutes=5)
    f_arr = pd.DatetimeIndex(features)
    for _, row in ds2.iterrows():
        ct = row['CME_DONKI_time']
        if pd.isna(ct) or not (features[0] <= ct <= features[-1]):
            continue
        if not (row['CME_DONKI_speed'] >= 0):
            continue
        pos = f_arr.searchsorted(ct + CME_OFFSET, side='left')
        if pos >= len(features):
            continue
        info = {a: row[a] for a in CME_ATTRS}
        info['cme_donki_time'] = ct
        for j in range(pos, min(pos + span, len(features))):
            cmes[features[j]] = info
    for key in CME_KEYS:
        data[key] = [cmes[t].get(key, 0) if isinstance(cmes[t], dict) else 0 for t in features]

    month, nine, nine_fast, maxspeed = cme_stats_vectorised(cme_sorted[0], cme_sorted[1], features)
    data['CMEs Past Month'] = month
    data['CMEs Past 9 Hours'] = nine
    data['CMEs Speed > 1000'] = nine_fast
    data['Max CME Speed'] = maxspeed

    return pd.DataFrame(data), onset, end


def interpolate_and_extrapolate(df, columns):
    for c in columns:
        if df[c].notna().any():
            df[c] = df[c].interpolate(method='linear', limit_direction='both')
        else:
            df[c] = df[c].fillna(-9999)
    return df


def finalize(df, sn_dict):
    # sunspot fill from SILSO
    miss = df['Sunspot Number'].isna()
    if miss.any():
        df.loc[miss, 'Sunspot Number'] = [
            sn_dict.get(pd.Timestamp(t).normalize(), np.nan)
            for t in df.loc[miss, 'Timestamp']
        ]
    # interpolate/extrapolate flux columns
    flux_cols = ['Proton Intensity']
    for _, prefix in CHANNELS:
        flux_cols += [f'{prefix}_tminus{i}' for i in range(24, 0, -1)] + [f'{prefix}_t']
    flux_cols += [f'p_tminus{i}' for i in range(24, 0, -1)] + ['p_t']
    df = interpolate_and_extrapolate(df, flux_cols)
    # recompute max intensities row-wise
    for _, prefix in CHANNELS:
        cols = [f'{prefix}_tminus{i}' for i in range(24, 0, -1)] + [f'{prefix}_t']
        df[f'{prefix}_max_intensity'] = df[cols].max(axis=1)
    pcols = [f'p_tminus{i}' for i in range(24, 0, -1)] + ['p_t']
    df['p_max_intensity'] = df[pcols].max(axis=1)
    # delta_log_Intensity
    df['delta_log_Intensity'] = np.log1p(df['Proton Intensity']) - np.log1p(df['p_t'])
    return df


def trim_background(df, threshold=0.1):
    """Same semantics as modules/training/utils.py:trim_background."""
    ts = pd.to_datetime(df['Target Timestamp'])
    lookup = dict(zip(ts, df['Proton Intensity']))
    for i, t in enumerate(ts):
        fut = lookup.get(t + pd.Timedelta(hours=3))
        if fut is not None and fut > threshold:
            return df.iloc[i:].reset_index(drop=True)
    return df


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    print("loading flux ...", flush=True)
    flux = load_flux_data(FLUX_PATH)
    print(f"  flux rows={len(flux)} {flux.index[0]} -> {flux.index[-1]}", flush=True)

    ds1 = pd.read_csv(SEP_PATH)
    ds1['datetime'] = pd.to_datetime(ds1['datetime'])
    ds1 = ds1[(ds1['Year'] >= 2010) & (ds1['Year'] <= 2017)]

    ds2 = pd.read_csv(CME_PATH)
    ds2['CME_DONKI_time'] = pd.to_datetime(ds2['CME_DONKI_time'], format='%m/%d/%Y %H:%M')
    ds2_daily = {}
    for _, r in ds2.iterrows():
        if pd.notna(r['CME_DONKI_time']):
            ds2_daily.setdefault(r['CME_DONKI_time'].date(), r['daily_sunspots'])

    valid = ds2.dropna(subset=['CME_DONKI_time']).sort_values('CME_DONKI_time')
    cme_sorted = (pd.DatetimeIndex(valid['CME_DONKI_time'].values),
                  valid['CME_DONKI_speed'].values.astype(float))

    sn = pd.read_csv(SN_PATH, delimiter=';', header=None,
                     names=['Year', 'Month', 'Day', 'Decimal Year', 'Sunspot Number',
                            'Standard Error', 'Observations', 'Definitive/Provisional'])
    sn['Date'] = pd.to_datetime(sn[['Year', 'Month', 'Day']])
    sn_dict = dict(zip(sn['Date'], sn['Sunspot Number']))

    groups = list(ds1.groupby(ds1.index // 4))
    print(f"catalog groups 2010-2017: {len(groups)}", flush=True)

    manifest = []
    for n, (_, g) in enumerate(groups, start=1):
        df, onset, end = build_event(n, g, ds2, flux, sn_dict, cme_sorted)
        df = finalize(df, sn_dict)
        df.to_csv(os.path.join(OUT, f"sep_event_{n}_filled_ie.csv"), index=False)
        tr = trim_background(df)
        tr.to_csv(os.path.join(OUT, f"sep_event_{n}_filled_ie_trim.csv"), index=False)
        manifest.append(dict(event=n, onset=onset, end=end, n_untrimmed=len(df),
                             n_trim=len(tr),
                             start_untrimmed=df['Target Timestamp'].iloc[0],
                             start_trim=tr['Target Timestamp'].iloc[0],
                             end_ts=df['Target Timestamp'].iloc[-1]))
        print(f"  event {n:>2}: untrimmed={len(df):>5} trim={len(tr):>5} "
              f"{df['Target Timestamp'].iloc[0]} -> {df['Target Timestamp'].iloc[-1]}", flush=True)

    pd.DataFrame(manifest).to_csv(os.path.join(OUT, "_manifest.csv"), index=False)
    print("done ->", OUT)
