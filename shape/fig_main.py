"""Paper 1 result figures, column width, at most two panels per row:
fig_2d.pdf  (a) existing 2D estimators vs object size, (b) the corrections on a zoomed scale;
fig_3d.pdf  sphericity bias vs slice thickness;
fig_nuclei.pdf  (a,b) circularity of real nuclei re-segmented at three resolutions, default vs F1.
Colours: one fixed hue per estimator, validated per panel (dataviz validate_palette.js)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from estimators import KAPPA

plt.rcParams.update({'font.size': 7, 'font.family': 'serif', 'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
                     'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
                     'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'legend.fontsize': 6.3,
                     'axes.titlesize': 7.2, 'axes.titleweight': 'bold'})
COL = {'default': '#e34948', 'ImageJ': '#6250d6', 'PyRadiomics': '#c98500',
       'Crofton': '#199e70', 'F1': '#eb6834', 'RCC': '#2a78d6'}
MK = {'default': 'o', 'ImageJ': '^', 'PyRadiomics': 'v', 'Crofton': 'x', 'F1': 's', 'RCC': 'P'}
LEG = dict(frameon=False, handlelength=1.6, labelspacing=0.2, borderaxespad=0.1, handletextpad=0.4)
OUT = '../paper/figures/'

# ---------- 2D synthetic ----------
df = pd.read_csv('exp1_synth_rc.csv')
COLS = {'default': ('P_skimage/CellProfiler', 'default'), 'ImageJ': ('P_ImageJ', 'ImageJ'),
        'PyRadiomics': ('P_PyRadiomics', 'PyRadiomics'), 'Crofton': ('P_Crofton (skimage)', 'Crofton'),
        'F1': ('P_corrected chain (ours)', 'F1 (ours)'), 'RCC': ('P_rc', 'RCC (ours)')}


def bias2d(ax, keys):
    for k in keys:
        col, lab = COLS[k]
        g = (4 * np.pi * df.Ad / df[col] ** 2 - df.ff_true).groupby(df.r).mean()
        ax.plot(g.index, g.values, marker=MK[k], ms=3, lw=1.2, color=COL[k], label=lab)
    ax.axhline(0, color='#aaa', lw=0.6, zorder=0)
    ax.set_xscale('log'); ax.set_xticks([2, 4, 8, 16, 32]); ax.set_xticklabels([2, 4, 8, 16, 32]); ax.minorticks_off()
    ax.set_xlabel('object radius (pixels)')


fig, (a, b) = plt.subplots(1, 2, figsize=(3.45, 1.55), gridspec_kw=dict(wspace=0.42))
bias2d(a, ['default', 'ImageJ', 'PyRadiomics'])
th = (df.ff_true * ((df.L / (KAPPA * (df.L - 2 * np.sqrt(2)))) ** 2 - 1)).groupby(df.r).mean()
r = th.index[th.index >= 3]
a.plot(r, th[r], color='k', ls=':', lw=1.1, label='model (2)', zorder=5)
a.set_ylim(-0.12, 0.6); a.set_ylabel('circularity bias'); a.set_title('(a) existing tools', loc='left')
bias2d(b, ['Crofton', 'F1', 'RCC'])
b.set_ylim(-0.025, 0.06); b.set_yticks([-0.02, 0, 0.02, 0.04, 0.06])
b.set_title(r'(b) corrections, $8\times$ zoom', loc='left')
a.legend(loc='upper right', bbox_to_anchor=(1.03, 1.02), **LEG)
b.legend(loc='upper right', **LEG)
fig.savefig(OUT + 'fig_2d.pdf', bbox_inches='tight'); fig.savefig('fig_2d.png', dpi=250, bbox_inches='tight')

# ---------- 3D synthetic ----------
d3 = pd.read_csv('exp3d_synth.csv'); d3 = d3[d3.R >= 5]
fig, c = plt.subplots(figsize=(3.45, 1.2))
for col, lab, k, ls in [('prad', 'PyRadiomics / IBSI', 'PyRadiomics', '-'),
                        ('prad_iso', 'PyRadiomics, resampled', 'PyRadiomics', '--'),
                        ('crofton', 'Crofton', 'Crofton', '-'), ('rcc', 'RCC (ours)', 'RCC', '-')]:
    g = (d3[col] - d3.true).groupby(d3.slice)
    c.plot(g.mean().index, g.mean().values, marker=MK[k], ms=3, lw=1.2, ls=ls, color=COL[k], label=lab,
           mfc='white' if ls == '--' else COL[k])
    c.fill_between(g.mean().index, g.quantile(0.1), g.quantile(0.9), color=COL[k], alpha=0.12, lw=0)
c.axhline(0, color='#aaa', lw=0.6, zorder=0)
c.set_xlabel('slice thickness (mm)'); c.set_ylabel('sphericity bias'); c.set_ylim(-0.23, 0.06)
c.set_yticks([0, -0.1, -0.2]); c.set_xticks([1, 2, 3, 4, 5]); c.set_xlim(0.8, 5.2)
c.legend(loc='lower center', bbox_to_anchor=(0.5, 0.98), ncol=2, columnspacing=1.0, **LEG)
fig.savefig(OUT + 'fig_3d.pdf', bbox_inches='tight'); fig.savefig('fig_3d.png', dpi=250, bbox_inches='tight')

# ---------- real nuclei ----------
Rr = pd.read_csv('reseg_bbbc_all.csv'); x = np.linspace(0.5, 1.45, 300)
shades = {1: '#b4b2a9', 2: '#6f6d66', 4: '#1f1e1c'}
labels = {1: 'native', 2: r'$\frac{1}{2}$ resolution', 4: r'$\frac{1}{4}$ resolution'}
fig, (d, e) = plt.subplots(1, 2, figsize=(3.45, 1.35), sharey=True, gridspec_kw=dict(wspace=0.12))
for ax, k, ttl in [(d, 'sk', '(a) default'), (e, 'ours', '(b) F1 (ours)')]:
    for f in [1, 2, 4]:
        v = Rr[f'FF_{k}_{f}'].values; v = v[(v > 0.3) & (v < 2)]
        ax.plot(x, gaussian_kde(v)(x), color=shades[f], lw=1.3, label=labels[f])
        ax.axvline(np.median(v), color=shades[f], lw=0.8, ls=':')
    ax.axvline(1, color='#999', lw=0.6, zorder=0)
    ax.text(1.01, 0.97, 'perfect\ncircle', transform=ax.get_xaxis_transform(), fontsize=5.8, color='#666', va='top')
    ax.set_xlim(0.5, 1.45); ax.set_xticks([0.6, 0.8, 1.0, 1.2, 1.4]); ax.set_xlabel('circularity (FF)')
    ax.set_title(ttl, loc='left')
d.set_ylabel('density'); d.set_yticks([])
h, l = d.get_legend_handles_labels()
fig.legend(h, l, loc='lower center', bbox_to_anchor=(0.52, 0.98), ncol=3, columnspacing=1.2, **LEG)
fig.savefig(OUT + 'fig_nuclei.pdf', bbox_inches='tight'); fig.savefig('fig_nuclei.png', dpi=250, bbox_inches='tight')
