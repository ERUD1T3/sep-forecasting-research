"""
Parser-robustness audit for SEP-EC-untrimmed.

Checks everything a downstream parser could trip over: encoding, line endings,
field counts, quoting, header agreement across files, per-column dtype purity,
timestamp format/consistency, duplicates, ordering, value domains, and numeric
pathologies (inf, overflow, whitespace, exotic literals).
"""
import csv, glob, io, os, re, sys, collections
import numpy as np
import pandas as pd

ROOT = "/Users/josiasmoukpe/Desktop/florida tech exit/sep-forecasting-research"
DIRS = [os.path.join(ROOT, 'SEP-EC-untrimmed', 'full'),
        os.path.join(ROOT, 'SEP-EC-untrimmed', 'extra_catalog_events')]

FAIL, WARN, OK = [], [], []


def rec(level, msg):
    (FAIL if level == 'FAIL' else WARN if level == 'WARN' else OK).append(msg)


US_TS = re.compile(r'^(\d{1,2})/(\d{1,2})/(\d{4}) (\d{1,2}):(\d{2})$')
NUM = re.compile(r'^-?(\d+\.?\d*([eE][+-]?\d+)?|\.\d+([eE][+-]?\d+)?)$')

files = []
for d in DIRS:
    files += sorted(glob.glob(os.path.join(d, '*.csv')))
print(f"auditing {len(files)} files\n")

headers = {}
col_kinds = collections.defaultdict(set)
total_rows = 0

for path in files:
    name = os.path.relpath(path, ROOT)
    raw = open(path, 'rb').read()

    # --- encoding / BOM / line endings / terminal newline ---
    try:
        text = raw.decode('utf-8')
    except UnicodeDecodeError as e:
        rec('FAIL', f"{name}: not valid UTF-8 ({e})")
        continue
    if raw.startswith(b'\xef\xbb\xbf'):
        rec('FAIL', f"{name}: has a UTF-8 BOM")
    if b'\r' in raw:
        rec('WARN', f"{name}: contains CR (line endings not pure LF)")
    if not raw.endswith(b'\n'):
        rec('WARN', f"{name}: no trailing newline on last line")
    if b'\x00' in raw:
        rec('FAIL', f"{name}: contains NUL bytes")
    non_ascii = set(ch for ch in text if ord(ch) > 127)
    if non_ascii:
        rec('WARN', f"{name}: non-ASCII characters {sorted(non_ascii)[:5]}")

    # --- strict csv parse: field counts, quoting ---
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        rec('FAIL', f"{name}: empty file")
        continue
    hdr = rows[0]
    headers[name] = tuple(hdr)
    ncol = len(hdr)
    if len(set(hdr)) != ncol:
        dup = [c for c, n in collections.Counter(hdr).items() if n > 1]
        rec('FAIL', f"{name}: duplicate column names {dup}")
    bad_width = [i for i, r in enumerate(rows[1:], start=2) if len(r) != ncol]
    if bad_width:
        rec('FAIL', f"{name}: {len(bad_width)} rows with wrong field count (first at line {bad_width[0]})")
    if '"' in text:
        rec('WARN', f"{name}: contains quote characters (embedded delimiters?)")
    body = rows[1:]
    total_rows += len(body)
    if not body:
        rec('FAIL', f"{name}: header only, no data rows")
        continue

    # --- per-column value-shape purity + whitespace ---
    idx = {c: i for i, c in enumerate(hdr)}
    for c, i in idx.items():
        vals = [r[i] for r in body]
        if any(v != v.strip() for v in vals):
            rec('FAIL', f"{name}: column '{c}' has leading/trailing whitespace")
        if any(v == '' for v in vals):
            rec('FAIL', f"{name}: column '{c}' has empty fields")
        for v in vals:
            if US_TS.match(v):
                col_kinds[c].add('timestamp')
            elif NUM.match(v):
                col_kinds[c].add('number')
            else:
                col_kinds[c].add(f'other:{v[:20]}')

    # --- domain / semantic checks ---
    df = pd.read_csv(path)
    ts = pd.to_datetime(df['Timestamp'], format='%m/%d/%Y %H:%M', errors='coerce')
    tt = pd.to_datetime(df['Target Timestamp'], format='%m/%d/%Y %H:%M', errors='coerce')
    if ts.isna().any() or tt.isna().any():
        rec('FAIL', f"{name}: timestamp(s) not parseable with %m/%d/%Y %H:%M")
    else:
        if not tt.is_monotonic_increasing:
            rec('FAIL', f"{name}: Target Timestamp not monotonically increasing")
        if tt.duplicated().any():
            rec('FAIL', f"{name}: duplicate Target Timestamp values")
        off = (tt - ts).unique()
        if len(off) != 1 or off[0] != pd.Timedelta(minutes=30):
            rec('FAIL', f"{name}: Target-minus-Timestamp offset not uniformly 30 min: {off[:3]}")
        step = tt.diff().dropna().unique()
        if len(step) != 1 or step[0] != pd.Timedelta(minutes=5):
            rec('FAIL', f"{name}: grid not uniformly 5 min: {step[:3]}")
    if df.duplicated().any():
        rec('FAIL', f"{name}: {int(df.duplicated().sum())} fully duplicated rows")
    if df['Event ID'].nunique() != 1:
        rec('FAIL', f"{name}: multiple Event ID values {df['Event ID'].unique()[:5]}")
    if 'is_reconstructed' in df.columns:
        bad = set(df['is_reconstructed'].unique()) - {0, 1, 2}
        if bad:
            rec('FAIL', f"{name}: unexpected is_reconstructed values {bad}")
        s = ''.join(map(str, df['is_reconstructed'].values))
        if not re.fullmatch(r'1*0*2*', s):
            rec('FAIL', f"{name}: is_reconstructed blocks interleaved")

    # --- numeric pathologies ---
    num = df.select_dtypes(include=[np.number])
    arr = num.to_numpy(dtype=np.float64, na_value=np.nan)
    if np.isinf(arr).any():
        rec('FAIL', f"{name}: infinite values present")
    if num.isna().to_numpy().any():
        cols = [c for c in num.columns if num[c].isna().any()]
        rec('FAIL', f"{name}: NaN in numeric columns {cols[:5]}")
    if (arr == -9999).any():
        rec('FAIL', f"{name}: -9999 sentinel present")
    mx = np.nanmax(np.abs(arr))
    if mx > 1e30:
        rec('FAIL', f"{name}: values above 1e30 (raw fill leaking through?) max={mx:.3g}")

# --- cross-file consistency ---
uniq = set(headers.values())
if len(uniq) == 1:
    rec('OK', f"all {len(headers)} files share one identical header ({len(next(iter(uniq)))} columns)")
else:
    groups = collections.defaultdict(list)
    for n, h in headers.items():
        groups[h].append(n)
    rec('FAIL', f"{len(uniq)} different headers across files")
    for h, ns in groups.items():
        rec('FAIL', f"   {len(ns)} file(s) e.g. {ns[0]} -> {len(h)} cols")

print("=== per-column value shapes (should be one kind per column) ===")
mixed = {c: k for c, k in col_kinds.items() if len(k) > 1}
for c, k in list(mixed.items())[:12]:
    print(f"  MIXED  {c}: {sorted(k)[:4]}")
if not mixed:
    print("  none — every column holds a single value shape")
print()

print(f"total data rows: {total_rows:,}\n")
for m in FAIL:
    print("FAIL  " + m)
for m in WARN:
    print("WARN  " + m)
for m in OK:
    print("OK    " + m)
print(f"\n{len(FAIL)} FAIL / {len(WARN)} WARN")
