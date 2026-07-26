"""
Plot the proton series in the space the network actually sees.

modules/training/ts_modeling.py :: load_file_data(apply_log=True) - the default -
applies np.log1p to the flux columns and to Proton Intensity, and the stored target
delta_log_Intensity is log1p(Proton Intensity) - log1p(p_t). So the model's view is
ln(1 + I), NOT log10(I).

The difference matters: ln(1+I) ~= I for I << 1, so everything below ~0.01 is
effectively linear in I and the quiet pre-onset background collapses towards zero.
On a log10 axis that same background spans several visible decades.
"""
import glob, os, re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

ROOT = "/Users/josiasmoukpe/Desktop/florida tech exit/sep-forecasting-research"
OUTDIR = os.path.join(ROOT, 'SEP-EC-untrimmed', 'figures')

C_ORIG, C_PRE, C_POST = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e3e2df'
SURF = '#fcfcfb'

HANDLES = [plt.Line2D([], [], color=C_PRE, lw=2.4, label='Restored before onset'),
           plt.Line2D([], [], color=C_ORIG, lw=2.4, label='Distributed SEP-EC (unchanged)'),
           plt.Line2D([], [], color=C_POST, lw=2.4, label='Restored after event end')]


def load(ev):
    d = pd.read_csv(os.path.join(ROOT, 'SEP-EC-untrimmed', 'full',
                                 f'sep_event_{ev}_filled_ie.csv'))
    d['t'] = pd.to_datetime(d['Target Timestamp'])
    d['pi'] = pd.to_numeric(d['Proton Intensity'])
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


def style(ax, mode):
    if mode == 'log10':
        ax.set_yscale('log')
        ax.grid(True, which='minor', color=GRID, lw=0.4, alpha=0.6, zorder=0)
    ax.grid(True, which='major', color=GRID, lw=0.8, zorder=0)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=3)


def panel(ax, ev, man, mode):
    d = load(ev)
    r = man[man.event == ev].iloc[0]
    sh = d[d['is_reconstructed'] == 0]
    ax.axvspan(sh['t'].iloc[0], sh['t'].iloc[-1], color=C_ORIG, alpha=0.055, lw=0, zorder=1)
    seg(ax, d, 1, C_PRE, mode)
    seg(ax, d, 2, C_POST, mode)
    seg(ax, d, 0, C_ORIG, mode, lw=2.2, z=3)
    style(ax, mode)
    bits = []
    if r.rows_pre:
        bits.append(f"+{r.hours_pre:.0f}h before")
    if r.rows_post:
        bits.append(f"+{r.hours_post:.0f}h after")
    ax.set_title(f"Event {ev}  ({sh['t'].iloc[0]:%Y-%m-%d})\n{' · '.join(bits) or 'unchanged'}",
                 fontsize=9, color=INK, pad=6, loc='left')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=4))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_xlabel('Time', fontsize=8.5, color=INK2)


YLAB = {'log1p': 'ln(1 + Proton Intensity)', 'log10': 'Flux (1/(cm^2 s sr MeV))'}


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    man = pd.read_csv(os.path.join(ROOT, 'SEP-EC-untrimmed', 'manifest.csv'))

    # ---- side-by-side: the same events in both spaces ----
    evs = [1, 35]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.4), facecolor=SURF)
    for col, ev in enumerate(evs):
        for row, mode in enumerate(('log10', 'log1p')):
            ax = axes[row, col]
            ax.set_facecolor(SURF)
            panel(ax, ev, man, mode)
            ax.set_ylabel(YLAB[mode], fontsize=8.5, color=INK2)
            tag = ('log10 axis — what the eye sees'
                   if mode == 'log10' else 'ln(1+I) — what the network sees')
            ax.set_title(ax.get_title() + f"\n{tag}", fontsize=9, color=INK, loc='left', pad=6)
    fig.legend(handles=HANDLES, loc='upper center', ncol=3, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, 0.972), labelcolor=INK)
    fig.suptitle('Same data, two spaces: log10 flux vs the network\'s ln(1+I)',
                 fontsize=14, color=INK, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join(OUTDIR, 'space_comparison_log10_vs_log1p.png')
    fig.savefig(p, dpi=150, facecolor=SURF)
    plt.close(fig)
    print(p)

    # ---- all events, network space, 3x3 pages ----
    all_ev = sorted(man.event.tolist())
    pages = [all_ev[i:i + 9] for i in range(0, len(all_ev), 9)]
    for page in pages:
        fig, axes = plt.subplots(3, 3, figsize=(14.5, 12.6), facecolor=SURF)
        for ax in axes.ravel():
            ax.set_facecolor(SURF)
            ax.axis('off')
        for ax, ev in zip(axes.ravel(), page):
            ax.axis('on')
            panel(ax, ev, man, 'log1p')
        for ax in axes[:, 0]:
            if ax.axison:
                ax.set_ylabel(YLAB['log1p'], fontsize=8.5, color=INK2)
        fig.legend(handles=HANDLES, loc='upper center', ncol=3, frameon=False,
                   fontsize=10, bbox_to_anchor=(0.5, 0.955), labelcolor=INK)
        fig.suptitle(f'SEP-EC untrimmed + extended — events {page[0]}-{page[-1]}  '
                     f'(network space)', fontsize=14, color=INK, y=0.99)
        fig.text(0.5, 0.968, 'ln(1 + Proton Intensity), linear axis — the transform '
                 'load_file_data(apply_log=True) applies before the model',
                 ha='center', fontsize=9, color=INK2)
        fig.tight_layout(rect=[0, 0, 1, 0.925])
        pp = os.path.join(OUTDIR, f'events_{page[0]:02d}-{page[-1]:02d}_log1p.png')
        fig.savefig(pp, dpi=140, facecolor=SURF)
        plt.close(fig)
        print(pp)


if __name__ == '__main__':
    main()
