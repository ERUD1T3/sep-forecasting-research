"""Phase 5 — figures for SEP-EC-24h, in both log10 flux space and the network's ln(1+I)."""
import glob, os, re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

ROOT = "/Users/josiasmoukpe/Desktop/florida tech exit/sep-forecasting-research"
import sys as _s
VERSION = _s.argv[1] if len(_s.argv) > 1 else 'safe'
DATA = os.path.join(ROOT, f'SEP-EC-24h-{VERSION}', 'full')
OUTDIR = os.path.join(ROOT, f'SEP-EC-24h-{VERSION}', 'figures')
TARGET = 'p16.4_tplus6'

C_ORIG, C_PRE, C_POST = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e3e2df', '#fcfcfb'
HANDLES = [plt.Line2D([], [], color=C_PRE, lw=2.4, label='Restored before onset (24 h)'),
           plt.Line2D([], [], color=C_ORIG, lw=2.4, label='Distributed SEP-EC (unchanged)'),
           plt.Line2D([], [], color=C_POST, lw=2.4, label='Restored after event end (24 h)')]
YLAB = {'log1p': 'ln(1 + p16.4 intensity)', 'log10': 'Flux (1/(cm^2 s sr MeV))'}


def load(ev):
    d = pd.read_csv(os.path.join(DATA, f'sep_event_{ev}_24h.csv'))
    d['t'] = pd.to_datetime(d['Target Timestamp'], format='%m/%d/%Y %H:%M')
    d['pi'] = pd.to_numeric(d[TARGET])
    return d


def seg(ax, d, flag, color, mode, lw=2.0, z=2):
    m = d['is_reconstructed'].values == flag
    if not m.any():
        return
    idx = np.where(m)[0]
    lo, hi = idx.min(), idx.max()
    if flag != 0:
        lo, hi = max(lo - 1, 0), min(hi + 1, len(d) - 1)
    s = d.iloc[lo:hi + 1]
    y = np.log1p(s['pi']) if mode == 'log1p' else s['pi'].where(s['pi'] > 0)
    ax.plot(s['t'], y, color=color, lw=lw, zorder=z, solid_capstyle='round')


def panel(ax, ev, man, mode):
    d = load(ev)
    r = man[man.event == ev].iloc[0]
    sh = d[d['is_reconstructed'] == 0]
    ax.axvspan(sh['t'].iloc[0], sh['t'].iloc[-1], color=C_ORIG, alpha=0.055, lw=0, zorder=1)
    seg(ax, d, 1, C_PRE, mode)
    seg(ax, d, 2, C_POST, mode)
    seg(ax, d, 0, C_ORIG, mode, lw=2.2, z=3)
    if mode == 'log10':
        ax.set_yscale('log')
        ax.grid(True, which='minor', color=GRID, lw=0.4, alpha=0.6, zorder=0)
    ax.grid(True, which='major', color=GRID, lw=0.8, zorder=0)
    for s_ in ('top', 'right'):
        ax.spines[s_].set_visible(False)
    for s_ in ('left', 'bottom'):
        ax.spines[s_].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=3)
    # label the ACTUAL window relative to onset / event end - that is what was asked
    # for - not the size of the block appended to SEP-EC's own span
    hb, ha = r.hours_before_onset, r.hours_after_end
    ax.set_title(f"Event {ev}  ({sh['t'].iloc[0]:%Y-%m-%d})\n"
                 f"{hb:.0f}h before onset · {ha:.0f}h after end",
                 fontsize=9, color=INK, pad=6, loc='left')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=4))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_xlabel('Time', fontsize=8.5, color=INK2)


def pages(man, mode):
    evs = sorted(man.event.tolist())
    for page in [evs[i:i + 9] for i in range(0, len(evs), 9)]:
        fig, axes = plt.subplots(3, 3, figsize=(14.5, 12.6), facecolor=SURF)
        for ax in axes.ravel():
            ax.set_facecolor(SURF)
            ax.axis('off')
        for ax, ev in zip(axes.ravel(), page):
            ax.axis('on')
            panel(ax, ev, man, mode)
        for ax in axes[:, 0]:
            if ax.axison:
                ax.set_ylabel(YLAB[mode], fontsize=8.5, color=INK2)
        fig.legend(handles=HANDLES, loc='upper center', ncol=3, frameon=False,
                   fontsize=10, bbox_to_anchor=(0.5, 0.955), labelcolor=INK)
        fig.suptitle(f'SEP-EC 24h [{VERSION}] — events {page[0]}-{page[-1]}', fontsize=14, color=INK, y=0.99)
        sub = ('p16.4 target channel (EPHIN 16.40 MeV protons at t+6 = +30 min), '
               '1/(cm^2 s sr MeV), log scale  ·  gaps = zero flux' if mode == 'log10'
               else 'ln(1 + p16.4 intensity) — the transform load_file_data(apply_log=True) applies')
        fig.text(0.5, 0.968, sub + '  ·  shaded band = span of the distributed dataset',
                 ha='center', fontsize=9, color=INK2)
        fig.tight_layout(rect=[0, 0, 1, 0.925])
        p = os.path.join(OUTDIR, f'{mode}_events_{page[0]:02d}-{page[-1]:02d}.png')
        fig.savefig(p, dpi=140, facecolor=SURF)
        plt.close(fig)
        print(p)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    for f in glob.glob(os.path.join(OUTDIR, '*.png')):
        os.remove(f)
    man = pd.read_csv(os.path.join(ROOT, f'SEP-EC-24h-{VERSION}', 'manifest.csv'))
    for mode in ('log10', 'log1p'):
        pages(man, mode)
    # side-by-side detail
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.4), facecolor=SURF)
    for col, ev in enumerate([1, 35]):
        for row, mode in enumerate(('log10', 'log1p')):
            ax = axes[row, col]
            ax.set_facecolor(SURF)
            panel(ax, ev, man, mode)
            ax.set_ylabel(YLAB[mode], fontsize=8.5, color=INK2)
            ax.set_title(ax.get_title() + ('\nlog10 axis' if mode == 'log10'
                                           else '\nln(1+I) — network space'),
                         fontsize=9, color=INK, loc='left', pad=6)
    fig.legend(handles=HANDLES, loc='upper center', ncol=3, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, 0.972), labelcolor=INK)
    fig.suptitle("SEP-EC 24h — same events in both spaces", fontsize=14, color=INK, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join(OUTDIR, 'space_comparison_log10_vs_log1p.png')
    fig.savefig(p, dpi=150, facecolor=SURF)
    plt.close(fig)
    print(p)


if __name__ == '__main__':
    main()
