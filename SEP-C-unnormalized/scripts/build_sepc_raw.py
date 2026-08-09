"""
SEP-C unnormalized, row-aligned to the distributed files.

Philip's ask: the SEP-C features are min-max normalized, so a 1000 km/s CME reads as
0.34 and plotting features against predictions is meaningless; and there are no
timestamps to look an event up.

An unnormalized table already exists (misc/sep_10mev_full_raw.csv) but it is in source
order, while his predictions are indexed by the stratified/split training, testing and
fold files. Nothing links the two. This script recovers the mapping by reproducing the
exact normalization from the source and matching each distributed row back by its
feature fingerprint, then emits a raw file per distributed file with the SAME row count
and SAME row order, so prediction i lines up with raw row i positionally.
"""
import glob, os
import numpy as np
import pandas as pd

ROOT = "/Users/josiasmoukpe/Desktop/florida tech exit/sep-forecasting-research"
SRC = os.path.join(ROOT, 'SEP10MeV_Features_v2.csv')
SEPC = os.path.expanduser('~/Downloads/CISIR-data/SEP-C')
OUT = os.path.join(ROOT, 'SEP-C-unnormalized')

# raw columns to carry through, timestamps first so they are easy to find
TIMESTAMPS = ['SEP_onset_time', 'CME_DONKI_time', 'CME_CDAW_time']
RAW_FEATURES = ['solar_wind_speed', 'connection_angle_degrees', 'VlogV', 'CME_DONKI_speed',
                'CME_DONKI_latitude', 'CME_DONKI_longitude', 'CME_CDAW_MPA',
                'CME_CDAW_LinearSpeed', 'DONKI_half_width', 'Accelaration',
                '2nd_order_speed_final', '2nd_order_speed_20R', 'CPA', 'daily_sunspots',
                'CMEs_in_past_month', 'CMEs_in_past_9hours',
                'CMEs_with_speed_over_1000_in_past_9hours', 'max_CME_speed_in_past_day',
                'half_richardson_value', 'diffusive_shock', 'Type2_Viz_Area', 'Halo']
TARGETS = ['peak_intensity', 'ln_peak_intensity']

MM_FEATURES = ['VlogV', 'CME_DONKI_speed', 'CME_DONKI_latitude', 'CME_DONKI_longitude',
               'CME_CDAW_MPA', 'CME_CDAW_LinearSpeed', 'DONKI_half_width', 'Accelaration',
               '2nd_order_speed_final', '2nd_order_speed_20R', 'CPA', 'daily_sunspots',
               'CMEs_in_past_month', 'CMEs_in_past_9hours',
               'CMEs_with_speed_over_1000_in_past_9hours', 'max_CME_speed_in_past_day']


def min_max(c):
    lo, hi = c.min(), c.max()
    return pd.Series(np.zeros(len(c)), index=c.index) if lo == hi else (c - lo) / (hi - lo)


def normalized_from_source(df):
    """Reproduce ds_prep_sep_cme.ipynb :: preprocess_cme_features exactly."""
    p = {'ln_peak_intensity': df['ln_peak_intensity'],
         'solar_wind_speed_norm': min_max(df['solar_wind_speed']),
         'connection_angle_degrees_norm': min_max(df['connection_angle_degrees'])}
    lhr = np.log1p(-df['half_richardson_value'])
    lds = np.log1p(df['diffusive_shock'])
    lt2 = df['Type2_Viz_Area'].apply(lambda x: np.log(x) if x != 0 else np.log(1))
    for f in MM_FEATURES:
        p[f + '_norm'] = min_max(df[f])
    p['log_richardson_value_norm'] = min_max(lhr)
    p['log_diffusive_shock_norm'] = min_max(lds)
    p['log_Type2_Viz_Area_norm'] = min_max(lt2)
    p['Halo'] = df['Halo']
    return pd.DataFrame(p)


def fingerprint(d, cols):
    return [tuple(np.round(r.astype(float), 10)) for r in d[cols].values]


def main():
    os.makedirs(OUT, exist_ok=True)
    src = pd.read_csv(SRC)
    norm = normalized_from_source(src)

    files = (sorted(glob.glob(os.path.join(SEPC, '*.csv')))
             + sorted(glob.glob(os.path.join(SEPC, 'fold*', '*.csv')))
             + sorted(glob.glob(os.path.join(SEPC, 'misc', 'sep_10mev_full.csv'))))

    cols = list(pd.read_csv(files[0], nrows=0).columns)
    lookup = {}
    for i, k in enumerate(fingerprint(norm, cols)):
        lookup.setdefault(k, []).append(i)

    report = []
    for f in files:
        rel = os.path.relpath(f, SEPC)
        his = pd.read_csv(f)
        idx, ambiguous, missing = [], 0, 0
        for k in fingerprint(his, cols):
            hits = lookup.get(k, [])
            if not hits:
                idx.append(None); missing += 1
            else:
                if len(hits) > 1:
                    ambiguous += 1
                idx.append(hits[0])
        ok = missing == 0
        out = src.iloc[[i for i in idx if i is not None]].reset_index(drop=True)
        keep = TIMESTAMPS + RAW_FEATURES + TARGETS
        out = out[[c for c in keep if c in out.columns]]
        out.insert(0, 'source_row', [i for i in idx if i is not None])

        # verification: re-normalize the picked rows and compare to his values
        chk = normalized_from_source(src).iloc[[i for i in idx if i is not None]].reset_index(drop=True)
        same = np.allclose(chk[cols].astype(float).values,
                           his[cols].astype(float).values, rtol=1e-9, atol=1e-12)

        dst = os.path.join(OUT, rel.replace('.csv', '_raw.csv').replace('/', '_'))
        out.to_csv(dst, index=False)
        report.append(dict(file=rel, rows=len(his), matched=len(his) - missing,
                           ambiguous=ambiguous, values_verify=bool(same),
                           output=os.path.basename(dst)))
        print(f"  {rel:<44} {len(his):>5} rows  matched={len(his)-missing:>5}  "
              f"ambiguous={ambiguous:>3}  verified={same}")

    rep = pd.DataFrame(report)
    rep.to_csv(os.path.join(OUT, 'alignment_report.csv'), index=False)
    print(f"\nfiles written: {len(rep)}   all rows matched: "
          f"{bool((rep.rows == rep.matched).all())}   all values verified: "
          f"{bool(rep.values_verify.all())}")


if __name__ == '__main__':
    main()
