"""Plot distributed SEP-EC vs the reconstructed untrimmed+extended version."""
import glob, os, re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

ROOT = "/Users/josiasmoukpe/Desktop/florida tech exit/sep-forecasting-research"
OUTDIR = os.path.join(ROOT, 'SEP-EC-untrimmed', 'figures')

# validated categorical slots 1-3 (light mode), all-pairs clean
C_ORIG = '#2a78d6'   # original SEP-EC rows
C_PRE = '#eb6834'    # restored before onset
C_POST = '#1baf7a'   # restored after event end
INK = '#0b0b0b'
INK2 = '#52514e'
GRID = '#e3e2df'


def load(ev):
    d = pd.read_csv(os.path.join(ROOT, 'SEP-EC-untrimmed', 'full',
                                 f'sep_event_{ev}_filled_ie.csv'))
    d['t'] = pd.to_datetime(d['Target Timestamp'])
    d['pi'] = pd.to_numeric(d['Proton Intensity'])
    return d


def seg(ax, d, flag, color, lw=2.0, z=2):
    """Draw one provenance segment, bridging one row into its neighbours."""
    m = d['is_reconstructed'].values == flag
    if not m.any():
        return
    idx = np.where(m)[0]
    lo, hi = idx.min(), idx.max()
    lo = max(lo - 1, 0) if flag != 0 else lo
    hi = min(hi + 1, len(d) - 1) if flag != 0 else hi
    s = d.iloc[lo:hi + 1]
    # zeros are non-detections, not small values: leave them as gaps rather than
    # clamping to a floor (which would both invent a value and lift genuine
    # sub-floor readings upward)
    y = s['pi'].where(s['pi'] > 0)
    ax.plot(s['t'], y, color=color, lw=lw, zorder=z, solid_capstyle='round')


def style(ax):
    ax.set_yscale('log')
    ax.grid(True, which='major', color=GRID, lw=0.8, zorder=0)
    ax.grid(True, which='minor', color=GRID, lw=0.4, alpha=0.6, zorder=0)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=3)


def panel(ax, ev, man):
    d = load(ev)
    r = man[man.event == ev].iloc[0]
    shipped = d[d['is_reconstructed'] == 0]
    # shade the span the distributed dataset covers
    ax.axvspan(shipped['t'].iloc[0], shipped['t'].iloc[-1],
               color=C_ORIG, alpha=0.055, lw=0, zorder=1)
    seg(ax, d, 1, C_PRE)
    seg(ax, d, 2, C_POST)
    seg(ax, d, 0, C_ORIG, lw=2.2, z=3)
    style(ax)
    extra = []
    if r.rows_pre:
        extra.append(f"+{r.hours_pre:.0f}h before")
    if r.rows_post:
        extra.append(f"+{r.hours_post:.0f}h after")
    tag = " · ".join(extra) if extra else "unchanged"
    ax.set_title(f"Event {ev}  ({shipped['t'].iloc[0]:%Y-%m-%d})\n{tag}",
                 fontsize=9, color=INK, pad=6, loc='left')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=4))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_xlabel('Time', fontsize=8.5, color=INK2)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    man = pd.read_csv(os.path.join(ROOT, 'SEP-EC-untrimmed', 'manifest.csv'))

    # ---------- figure 1: small multiples ----------
    evs = [1, 5, 7, 9, 10, 16, 19, 25, 27, 33, 35, 43]
    fig, axes = plt.subplots(4, 3, figsize=(13, 14.6), facecolor='#fcfcfb')
    for ax, ev in zip(axes.ravel(), evs):
        ax.set_facecolor('#fcfcfb')
        panel(ax, ev, man)
    for ax in axes[:, 0]:
        ax.set_ylabel('Flux (1/(cm^2 s sr MeV))', fontsize=8.5, color=INK2)
    handles = [plt.Line2D([], [], color=C_PRE, lw=2.4, label='Restored before onset'),
               plt.Line2D([], [], color=C_ORIG, lw=2.4, label='Distributed SEP-EC (unchanged)'),
               plt.Line2D([], [], color=C_POST, lw=2.4, label='Restored after event end')]
    fig.legend(handles=handles, loc='upper center', ncol=3, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, 0.951), labelcolor=INK)
    fig.suptitle('SEP-EC: distributed vs reconstructed untrimmed + extended',
                 fontsize=14, color=INK, y=0.986, x=0.5)
    fig.text(0.5, 0.966, 'Proton Intensity (16.4 MeV channel), log scale  ·  shaded band = span of the distributed dataset',
             ha='center', fontsize=9, color=INK2)
    fig.tight_layout(rect=[0, 0, 1, 0.928])
    p1 = os.path.join(OUTDIR, 'overview_sample_12_events.png')
    fig.savefig(p1, dpi=150, facecolor='#fcfcfb')
    plt.close(fig)

    # ---------- figure 2: detail on two events ----------
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), facecolor='#fcfcfb')
    for ax, ev in zip(axes, [1, 35]):
        ax.set_facecolor('#fcfcfb')
        d = load(ev)
        r = man[man.event == ev].iloc[0]
        shipped = d[d['is_reconstructed'] == 0]
        ax.axvspan(shipped['t'].iloc[0], shipped['t'].iloc[-1],
                   color=C_ORIG, alpha=0.055, lw=0, zorder=1)
        seg(ax, d, 1, C_PRE, lw=2.4)
        seg(ax, d, 2, C_POST, lw=2.4)
        seg(ax, d, 0, C_ORIG, lw=2.6, z=3)
        for x, lab in ((shipped['t'].iloc[0], 'SEP-EC starts'),
                       (shipped['t'].iloc[-1], 'SEP-EC ends')):
            ax.axvline(x, color=INK2, lw=1, ls=(0, (4, 3)), zorder=4)
            ax.annotate(lab, xy=(x, 0.02), xycoords=('data', 'axes fraction'),
                        rotation=90, fontsize=8, color=INK2,
                        ha='right', va='bottom', xytext=(-3, 0),
                        textcoords='offset points')
        style(ax)
        ax.set_title(f"Event {ev} — {shipped['t'].iloc[0]:%Y-%m-%d} · "
                     f"{len(shipped)} distributed rows → {len(d)} rows",
                     fontsize=10.5, color=INK, loc='left', pad=8)
        ax.set_ylabel('Flux (1/(cm^2 s sr MeV))', fontsize=8.5, color=INK2)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=5))
        plt.setp(ax.get_xticklabels(), rotation=20, ha='right')
    fig.legend(handles=handles, loc='upper center', ncol=3, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, 1.0), labelcolor=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    p2 = os.path.join(OUTDIR, 'detail_events_01_and_35.png')
    fig.savefig(p2, dpi=150, facecolor='#fcfcfb')
    plt.close(fig)
    # ---------- figures 3+: every event, 3x3 pages ----------
    evs_all = sorted(man.event.tolist())
    pages = [evs_all[i:i + 9] for i in range(0, len(evs_all), 9)]
    for pi_, page in enumerate(pages, start=1):
        fig, axes = plt.subplots(3, 3, figsize=(14.5, 12.6), facecolor='#fcfcfb')
        for ax in axes.ravel():
            ax.set_facecolor('#fcfcfb')
            ax.axis('off')
        for ax, ev in zip(axes.ravel(), page):
            ax.axis('on')
            panel(ax, ev, man)
        for ax in axes[:, 0]:
            if ax.axison:
                ax.set_ylabel('Flux (1/(cm^2 s sr MeV))', fontsize=8.5, color=INK2)
        fig.legend(handles=handles, loc='upper center', ncol=3, frameon=False,
                   fontsize=10, bbox_to_anchor=(0.5, 0.955), labelcolor=INK)
        fig.suptitle(f'SEP-EC untrimmed + extended — events {page[0]}-{page[-1]}',
                     fontsize=14, color=INK, y=0.99)
        fig.text(0.5, 0.968, 'Proton Intensity — EPHIN 16.40 MeV proton channel, 1/(cm^2 s sr MeV), log scale  ·  '
                 'shaded band = span of the distributed dataset  ·  gaps = zero flux',
                 ha='center', fontsize=9, color=INK2)
        fig.tight_layout(rect=[0, 0, 1, 0.925])
        pp = os.path.join(OUTDIR, f'events_{page[0]:02d}-{page[-1]:02d}.png')
        fig.savefig(pp, dpi=140, facecolor='#fcfcfb')
        plt.close(fig)
        print(pp)

    print(p1)
    print(p2)


if __name__ == '__main__':
    main()
