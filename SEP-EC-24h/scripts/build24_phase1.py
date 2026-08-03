"""
Phase 1 — regenerate all 46 catalog events from the recovered ephin5m_v2.dat,
with the window widened to onset-24h ... catalog_end+24h.

Differences from the original pipeline, all deliberate:
  * flux source is the real ephin5m_v2.dat (2012 present, 72.4% covered)
  * HOURS_BEFORE 12 -> 24, plus HOURS_AFTER 24 (the original stopped at catalog end)
  * CME attachment (Philip's rule 1d: attach at CME_time+30min, persist
    avg_sep_duration) now runs over the FULL widened window, so CMEs that began
    before the old 12h boundary are no longer invisible
  * every row carries flux_raw_backed: was there a genuine flux datum at that
    timestamp BEFORE interpolation
"""
import os
from datetime import timedelta

import numpy as np
import pandas as pd

ROOT = "/Users/josiasmoukpe/Desktop/florida tech exit/sep-forecasting-research"
OUT = os.path.join(ROOT, "rebuild24")
FLUX_PATH = os.path.join(ROOT, "ephin5m_v2.dat")
SEP_PATH = os.path.join(ROOT, "curr_pf10th10_original.csv")
CME_PATH = os.path.join(ROOT, "SEP10MeV_Features_v2.csv")
SN_PATH = os.path.join(ROOT, "SN_d_tot_V2.0.csv")

HOURS_BEFORE = 24
HOURS_AFTER = 24
CME_OFFSET = timedelta(minutes=30)
AVG_SEP_DURATION = timedelta(days=0, hours=20, minutes=13, seconds=56, microseconds=86956)
PROTON_CHANNEL = "Proton_Flux_16.40MeV"

# (raw column, output prefix) — p16.4 now carries its energy, per Philip's 1c
CHANNELS = [
    ("Electron_Flux_0.5MeV", "e0.5"),
    ("Electron_Flux_1.8MeV", "e1.8"),
    ("Electron_Flux_4.4MeV", "e4.4"),
    ("Proton_Flux_6.10MeV", "p6.1"),
    ("Proton_Flux_33.00MeV", "p33.0"),
]
P_PREFIX = "p16.4"
TARGET_COL = "p16.4_tplus6"          # was 'Proton Intensity' (column D)

CME_ATTRS = ['CME_DONKI_latitude', 'CME_DONKI_longitude', 'CME_DONKI_speed',
             'CME_CDAW_MPA', 'CME_CDAW_LinearSpeed', 'VlogV', 'DONKI_half_width',
             'Accelaration', '2nd_order_speed_final', '2nd_order_speed_20R',
             'CPA', 'Halo', 'Type2_Viz_Area', 'solar_wind_speed', 'diffusive_shock',
             'half_richardson_value']
CME_KEYS = ['cme_donki_time'] + CME_ATTRS


def load_flux_data(filepath):
    df = pd.read_csv(filepath, sep=r"\s+", header=None, skiprows=4)
    df.columns = ['Year', 'Seconds_of_year', 'Electron_Flux_0.5MeV', 'Electron_Flux_1.8MeV',
                  'Electron_Flux_4.4MeV', 'Electron_Flux_7.4MeV', 'Proton_Flux_0.54MeV',
                  'Proton_Flux_1.37MeV', 'Proton_Flux_4.01MeV', 'Proton_Flux_6.10MeV',
                  'Proton_Flux_16.40MeV', 'Proton_Flux_33.00MeV', 'Proton_Flux_47.00MeV']
    df['Time'] = (pd.to_datetime(df['Year'].astype(int).astype(str), format='%Y')
                  + pd.to_timedelta(df['Seconds_of_year'].astype('int64'), unit='s'))
    df = df.set_index('Time')
    return df.replace(-9.9999998E+30, np.nan)


def nearest(fidx, values, times):
    return values[fidx.get_indexer(pd.DatetimeIndex(times), method='nearest')]


def build_time_grid(group):
    onset = group[group['Index'] == 1]['datetime'].iloc[0]
    peak = group[group['Index'] == 3]['datetime'].iloc[0]
    end = group[group['Index'] == 4]['datetime'].iloc[0]
    start = onset - timedelta(hours=HOURS_BEFORE)
    end_ext = end + timedelta(hours=HOURS_AFTER)
    targets, cur = [], start
    while cur <= end_ext:
        targets.append(cur)
        cur += timedelta(minutes=5)
    if targets[-1] != end_ext:
        targets.append(cur)
    features = [t - timedelta(minutes=30) for t in targets]
    return onset, peak, start, end, end_ext, targets, features


def cme_stats(cme_times, cme_speeds, features):
    """current > t >= current - H ; speed strictly > 1000."""
    f = pd.DatetimeIndex(features)

    def win(h):
        hi = cme_times.searchsorted(f, side='left')
        lo = cme_times.searchsorted(f - pd.Timedelta(hours=h), side='left')
        return lo, hi

    lo_m, hi_m = win(30 * 24)
    lo_9, hi_9 = win(9)
    fast = np.cumsum(np.concatenate([[0], (cme_speeds > 1000).astype(int)]))
    lo_d, hi_d = win(24)
    mx = np.zeros(len(f))
    for i in range(len(f)):
        if hi_d[i] > lo_d[i]:
            seg = cme_speeds[lo_d[i]:hi_d[i]]
            mx[i] = np.nanmax(seg) if np.isfinite(seg).any() else 0
    return hi_m - lo_m, hi_9 - lo_9, fast[hi_9] - fast[lo_9], mx


def build_event(event_id, group, ds2, flux, sn_dict, ds2_daily, cme_sorted, raw_ok):
    onset, peak, start, end, end_ext, targets, features = build_time_grid(group)
    fidx = flux.index

    data = {
        'Event ID': [event_id] * len(features),
        'Timestamp': features,
        'Target Timestamp': targets,
        TARGET_COL: nearest(fidx, flux[PROTON_CHANNEL].values, targets),
    }

    for channel, prefix in CHANNELS:
        vals = flux[channel].values
        for i in range(24, 0, -1):
            data[f'{prefix}_tminus{i}'] = nearest(
                fidx, vals, [t - timedelta(minutes=5 * i) for t in features])
        data[f'{prefix}_t'] = nearest(fidx, vals, features)
        data[f'{prefix}_max_intensity'] = np.zeros(len(features))

    pvals = flux[PROTON_CHANNEL].values
    for i in range(24, 0, -1):
        data[f'{P_PREFIX}_tminus{i}'] = nearest(
            fidx, pvals, [t - timedelta(minutes=5 * i) for t in features])
    data[f'{P_PREFIX}_t'] = nearest(fidx, pvals, features)
    data[f'{P_PREFIX}_max_intensity'] = np.zeros(len(features))

    data['Sunspot Number'] = [ds2_daily.get(t.date(), np.nan) for t in features]

    # --- CME attachment over the FULL widened window (Philip 1d) ---
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

    m, n9, nf, mx = cme_stats(cme_sorted[0], cme_sorted[1], features)
    data['CMEs Past Month'] = m
    data['CMEs Past 9 Hours'] = n9
    data['CMEs Speed > 1000'] = nf
    data['Max CME Speed'] = mx

    df = pd.DataFrame(data)
    df['flux_raw_backed'] = raw_ok.values[
        fidx.get_indexer(pd.DatetimeIndex(df['Target Timestamp']), method='nearest')].astype(int)
    df['onset'] = onset
    df['catalog_end'] = end
    return df


def interpolate_and_extrapolate(df, columns):
    for c in columns:
        if df[c].notna().any():
            df[c] = df[c].interpolate(method='linear', limit_direction='both')
        else:
            df[c] = df[c].fillna(-9999)
    return df


def finalize(df, sn_dict):
    miss = df['Sunspot Number'].isna()
    if miss.any():
        df.loc[miss, 'Sunspot Number'] = [
            sn_dict.get(pd.Timestamp(t).normalize(), np.nan) for t in df.loc[miss, 'Timestamp']]
    prefixes = [p for _, p in CHANNELS] + [P_PREFIX]
    flux_cols = [TARGET_COL]
    for p in prefixes:
        flux_cols += [f'{p}_tminus{i}' for i in range(24, 0, -1)] + [f'{p}_t']
    df = interpolate_and_extrapolate(df, flux_cols)
    for p in prefixes:
        cols = [f'{p}_tminus{i}' for i in range(24, 0, -1)] + [f'{p}_t']
        df[f'{p}_max_intensity'] = df[cols].max(axis=1)
    df['delta_log_Intensity'] = np.log1p(df[TARGET_COL]) - np.log1p(df[f'{P_PREFIX}_t'])
    return df.copy()


def main():
    os.makedirs(OUT, exist_ok=True)
    print("loading flux (ephin5m_v2.dat) ...", flush=True)
    flux = load_flux_data(FLUX_PATH)
    raw_ok = flux[PROTON_CHANNEL].notna()
    print(f"  rows={len(flux):,}  {flux.index[0]} -> {flux.index[-1]}", flush=True)

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
    for n, (_, g) in enumerate(groups, start=1):
        df = build_event(n, g, ds2, flux, sn_dict, ds2_daily, cme_sorted, raw_ok)
        df = finalize(df, sn_dict)
        df.to_csv(os.path.join(OUT, f'sep_event_{n}_ext24.csv'), index=False)
        pre = (pd.to_datetime(df['Target Timestamp']) < df['onset'].iloc[0]).sum()
        post = (pd.to_datetime(df['Target Timestamp']) > df['catalog_end'].iloc[0]).sum()
        print(f"  event {n:>2}: rows={len(df):>5}  backed={100*df.flux_raw_backed.mean():5.1f}%"
              f"  pre24={pre:>4} post24={post:>4}", flush=True)
    print("done ->", OUT)


if __name__ == '__main__':
    main()
