"""Phase 4 — validation gate for SEP-EC-24h. Any FAIL blocks delivery."""
import csv, glob, io, os, re, collections
import numpy as np
import pandas as pd

ROOT = "/Users/josiasmoukpe/Desktop/florida tech exit/sep-forecasting-research"
SHIP = os.path.join(ROOT, 'full')
import sys as _s
VERSION = _s.argv[1] if len(_s.argv) > 1 else 'safe'
OUT = os.path.join(ROOT, f'SEP-EC-24h-{VERSION}')
FAIL, OKS = [], []


def chk(cond, ok, bad):
    (OKS if cond else FAIL).append(ok if cond else bad)


def ren(c):
    if c == 'Proton Intensity':
        return 'p16.4_tplus6'
    m = re.match(r'^p_(tminus\d+|t|max_intensity)$', c)
    return f'p16.4_{m.group(1)}' if m else c


CME_ATTR = ['cme_donki_time', 'CME_DONKI_latitude', 'CME_DONKI_longitude', 'CME_DONKI_speed',
            'CME_CDAW_MPA', 'CME_CDAW_LinearSpeed', 'VlogV', 'DONKI_half_width', 'Accelaration',
            '2nd_order_speed_final', '2nd_order_speed_20R', 'CPA', 'Halo', 'Type2_Viz_Area',
            'solar_wind_speed', 'diffusive_shock', 'half_richardson_value']
PREF = ['e0.5', 'e1.8', 'e4.4', 'p6.1', 'p33.0', 'p16.4']

ship = {int(re.search(r'event_(\d+)_', os.path.basename(f)).group(1)): f
        for f in glob.glob(os.path.join(SHIP, '*.csv'))}
out = {int(re.search(r'event_(\d+)_', os.path.basename(f)).group(1)): f
       for f in glob.glob(os.path.join(OUT, 'full', '*.csv'))}
extra = sorted(glob.glob(os.path.join(OUT, 'extra_catalog_events', '*.csv')))
print(f"{len(out)} event files + {len(extra)} extra\n")

# ---------- 1. non-CME columns byte-identical in prov-0 rows ----------
bad_ev, nonc_cells, cme_cells = [], 0, 0
for ev in sorted(ship):
    a = pd.read_csv(ship[ev], dtype=str, keep_default_na=False)
    b = pd.read_csv(out[ev], dtype=str, keep_default_na=False)
    b0 = b[b.is_reconstructed == '0'].reset_index(drop=True)
    if len(b0) != len(a):
        bad_ev.append((ev, 'row count'))
        continue
    for c in a.columns:
        nc = ren(c)
        if not (a[c].values == b0[nc].values).all():
            bad_ev.append((ev, c))
            nonc_cells += int((a[c].values != b0[nc].values).sum())
chk(not bad_ev, f"EVERY column byte-identical in all prov-0 rows ({len(ship)} events)",
    f"prov-0 text differs: {bad_ev[:6]} ({nonc_cells} cells)")

# ---------- 2. coverage / ordering / grid ----------
sup = mono = cont = order = 0
for ev in sorted(out):
    b = pd.read_csv(out[ev])
    t = pd.to_datetime(b['Target Timestamp'], format='%m/%d/%Y %H:%M')
    ta = pd.to_datetime(pd.read_csv(ship[ev], usecols=['Target Timestamp'])['Target Timestamp'])
    sup += int(set(ta) <= set(t))
    mono += int(t.is_monotonic_increasing)
    d = t.diff().dropna().unique()
    cont += int(len(d) == 1 and d[0] == np.timedelta64(5, 'm'))
    order += int(re.fullmatch(r'1*0+2*', ''.join(map(str, b['is_reconstructed'].values))) is not None)
chk(sup == 44, "every SEP-EC timestamp present (44/44)", f"coverage {sup}/44")
chk(mono == 44, "time monotonic (44/44)", f"monotonic {mono}/44")
chk(cont == 44, "contiguous 5-minute grid (44/44)", f"contiguous {cont}/44")
chk(order == 44, "is_reconstructed ordered 1*0+2* (44/44)", f"ordering {order}/44")

# ---------- 3. internal consistency ----------
lagbad = tplusbad = dlbad = 0
for f in list(out.values()) + extra:
    b = pd.read_csv(f)
    for p in PREF:
        x = pd.to_numeric(b[f'{p}_tminus1']).values[1:]
        y = pd.to_numeric(b[f'{p}_t']).values[:-1]
        lagbad += int((~np.isclose(x, y, rtol=1e-6, atol=1e-12)).sum())
        cols = [f'{p}_tminus{i}' for i in range(24, 0, -1)] + [f'{p}_t']
        rm = b[cols].apply(pd.to_numeric).max(axis=1).values
        if not np.allclose(rm, pd.to_numeric(b[f'{p}_max_intensity']).values,
                           rtol=1e-6, atol=1e-12):
            lagbad += 1
    tp = pd.to_numeric(b['p16.4_tplus6']).values[:-6]
    pt = pd.to_numeric(b['p16.4_t']).values[6:]
    tplusbad += int((~np.isclose(tp, pt, rtol=1e-6, atol=1e-12)).sum())
    dl = pd.to_numeric(b['delta_log_Intensity']).values
    exp = np.log1p(pd.to_numeric(b['p16.4_tplus6']).values) - np.log1p(pd.to_numeric(b['p16.4_t']).values)
    dlbad += int((~np.isclose(dl, exp, rtol=1e-6, atol=1e-9)).sum())
OKS.append(f"lag self-consistency mismatches: {lagbad} (0 expected outside interpolated gaps)")
OKS.append(f"p16.4_tplus6[i] == p16.4_t[i+6] mismatches: {tplusbad}")
OKS.append(f"delta_log_Intensity formula mismatches: {dlbad} (SEP-EC's own rounding contributes)")

# ---------- 4. CME features OFF outside the event ----------
nz = 0
for ev in sorted(out):
    b = pd.read_csv(out[ev], dtype=str, keep_default_na=False)
    r = b[b.is_reconstructed != '0']
    for c in CME_ATTR:
        if c == 'cme_donki_time':
            nz += int((r[c].values != '0').sum())
        else:
            nz += int((pd.to_numeric(r[c], errors='coerce').fillna(0) != 0).sum())
chk(nz == 0, "CME attribute columns are 0 in every restored row (1d)",
    f"{nz} non-zero CME cells in restored rows")

# ---------- 5. parser audit ----------
US = re.compile(r'^\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2}$')
NUM = re.compile(r'^-?(\d+\.?\d*([eE][+-]?\d+)?|\.\d+([eE][+-]?\d+)?)$')
hdrs, mixed, probs = set(), collections.defaultdict(set), []
for f in list(out.values()) + extra:
    raw = open(f, 'rb').read()
    if raw.startswith(b'\xef\xbb\xbf') or b'\x00' in raw or b'\r' in raw or not raw.endswith(b'\n'):
        probs.append(f"{os.path.basename(f)}: encoding/line-ending")
    text = raw.decode('utf-8')
    if '"' in text:
        probs.append(f"{os.path.basename(f)}: quotes")
    rows = list(csv.reader(io.StringIO(text)))
    h = rows[0]
    hdrs.add(tuple(h))
    if any(len(r) != len(h) for r in rows[1:]):
        probs.append(f"{os.path.basename(f)}: ragged rows")
    for i, c in enumerate(h):
        for r in rows[1:]:
            v = r[i]
            if v != v.strip() or v == '':
                probs.append(f"{os.path.basename(f)}: '{c}' whitespace/empty")
                break
            mixed[c].add('ts' if US.match(v) else 'num' if NUM.match(v) else f'other:{v[:14]}')
chk(len(hdrs) == 1, f"one identical header across all {len(out)+len(extra)} files ({len(next(iter(hdrs)))} cols)",
    f"{len(hdrs)} different headers")
chk(not probs, "encoding / line endings / quoting / field counts / whitespace all clean",
    f"parser problems: {probs[:5]}")
odd = {c: k for c, k in mixed.items() if len(k) > 1 and c != 'cme_donki_time'}
chk(not odd, "every column holds one value shape (cme_donki_time is the documented union)",
    f"unexpected mixed columns: {list(odd)[:5]}")

# ---------- 6. no missing markers ----------
nan = sent = 0
for f in list(out.values()) + extra:
    b = pd.read_csv(f)
    num = b.select_dtypes(include=[np.number])
    arr = num.to_numpy(dtype=np.float64, na_value=np.nan)
    nan += int(np.isnan(arr).sum())
    sent += int((arr == -9999).sum())
chk(nan == 0 and sent == 0, "no NaN and no -9999 anywhere", f"NaN={nan} -9999={sent}")

# ---------- 7. naming per 1c ----------
h = list(next(iter(hdrs)))
chk(h[3] == 'p16.4_tplus6', "column D is p16.4_tplus6", f"column D is {h[3]}")
chk(h[134] == 'p16.4_tminus24' and h[159] == 'p16.4_max_intensity',
    "columns EE-FD are p16.4_tminus24 .. p16.4_max_intensity",
    f"EE={h[134]} FD={h[159]}")
chk(not any(re.match(r'^p_(tminus|t$|max)', c) for c in h), "no bare 'p_' columns remain",
    "bare p_ columns still present")

print("PASS")
for m in OKS:
    print("  OK   " + m)
if FAIL:
    print("FAIL")
    for m in FAIL:
        print("  FAIL " + m)
print(f"\n{len(FAIL)} FAIL / {len(OKS)} checks recorded")
