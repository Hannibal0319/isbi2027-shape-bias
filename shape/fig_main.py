"""Paper 1, Figure 2. Top: (a) existing 2D estimators vs object size, (b) the corrections on a zoomed scale,
(c) 3D sphericity vs slice thickness. Bottom: (d,e) form factor of real nuclei re-segmented at three resolutions,
default estimator vs F1. Colours: one fixed hue per estimator, validated per panel (dataviz validate_palette.js)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from estimators import KAPPA

plt.rcParams.update({'font.size': 7.5, 'font.family': 'serif', 'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
                     'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
                     'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'legend.fontsize': 6.6,
                     'axes.titlesize': 7.6, 'axes.titleweight': 'bold'})
# existing tools: red / violet / dark yellow; corrections: aqua (Crofton), orange (F1), blue (RCC, ours)
COL = {'default': '#e34948', 'ImageJ': '#6250d6', 'PyRadiomics': '#c98500',
       'Crofton': '#199e70', 'F1': '#eb6834', 'RCC': '#2a78d6'}
MK = {'default': 'o', 'ImageJ': '^', 'PyRadiomics': 'v', 'Crofton': 'x', 'F1': 's', 'RCC': 'P'}
LEG = dict(frameon=False, handlelength=1.7, labelspacing=0.2, borderaxespad=0.1)

fig = plt.figure(figsize=(7.1, 2.75))
gs = fig.add_gridspec(2, 12, height_ratios=[1, 0.8], hspace=0.62, wspace=5.0)
ax_a, ax_b, ax_c = (fig.add_subplot(gs[0, i:i + 4]) for i in (0, 4, 8))
ax_d = fig.add_subplot(gs[1, 0:6]); ax_e = fig.add_subplot(gs[1, 6:12], sharey=ax_d)

df = pd.read_csv('exp1_synth_rc.csv')
COLS = {'default': ('P_skimage/CellProfiler', 'default (chain code)'),
        'ImageJ': ('P_ImageJ', 'ImageJ'), 'PyRadiomics': ('P_PyRadiomics', 'PyRadiomics / IBSI'),
        'Crofton': ('P_Crofton (skimage)', 'Crofton'), 'F1': ('P_corrected chain (ours)', 'F1, post hoc (ours)'),
        'RCC': ('P_rc', 'RCC (ours)')}


def bias2d(ax, keys):
    for k in keys:
        col, lab = COLS[k]
        g = (4 * np.pi * df.Ad / df[col] ** 2 - df.ff_true).groupby(df.r).mean()
        ax.plot(g.index, g.values, marker=MK[k], ms=3.2, lw=1.2, color=COL[k], label=lab)
    ax.axhline(0, color='#aaa', lw=0.6, zorder=0)
    ax.set_xscale('log'); ax.set_xticks([2, 4, 8, 16, 32]); ax.set_xticklabels([2, 4, 8, 16, 32])
    ax.minorticks_off(); ax.set_xlabel('object radius (pixels)')


# (a) existing estimators
bias2d(ax_a, ['default', 'ImageJ', 'PyRadiomics'])
th = (df.ff_true * ((df.L / (KAPPA * (df.L - 2 * np.sqrt(2)))) ** 2 - 1)).groupby(df.r).mean()
r = th.index[th.index >= 3]
ax_a.plot(r, th[r], color='k', ls=':', lw=1.1, label='model, Eq. (2)', zorder=5)
ax_a.set_ylim(-0.12, 0.6); ax_a.set_ylabel('circularity bias')
ax_a.set_title('(a) 2D, existing tools', loc='left')
ax_a.legend(loc='upper right', **LEG)

# (b) corrections, zoomed y-scale
bias2d(ax_b, ['Crofton', 'F1', 'RCC'])
ax_b.set_ylim(-0.025, 0.06); ax_b.set_yticks([-0.02, 0, 0.02, 0.04, 0.06])
ax_b.set_title(r'(b) 2D, corrections ($8\times$ zoom)', loc='left')
ax_b.legend(loc='upper right', **LEG)

# (c) 3D sphericity bias vs slice thickness
d3 = pd.read_csv('exp3d_synth.csv'); d3 = d3[d3.R >= 5]; c = ax_c
for col, lab, k, ls in [('prad', 'PyRadiomics / IBSI', 'PyRadiomics', '-'),
                        ('prad_iso', 'PyRadiomics, resampled', 'PyRadiomics', '--'),
                        ('crofton', 'Crofton', 'Crofton', '-'), ('rcc', 'RCC (ours)', 'RCC', '-')]:
    g = (d3[col] - d3.true).groupby(d3.slice)
    c.plot(g.mean().index, g.mean().values, marker=MK[k], ms=3.2, lw=1.2, ls=ls, color=COL[k], label=lab,
           mfc='white' if ls == '--' else COL[k])
    c.fill_between(g.mean().index, g.quantile(0.1), g.quantile(0.9), color=COL[k], alpha=0.12, lw=0)
c.axhline(0, color='#aaa', lw=0.6, zorder=0)
c.set_xlabel('slice thickness (mm)'); c.set_ylabel('sphericity bias'); c.set_ylim(-0.24, 0.17)
c.set_yticks([0, -0.1, -0.2]); c.set_xticks([1, 2, 3, 4, 5]); c.set_title('(c) 3D sphericity', loc='left')
c.legend(loc='upper center', ncol=2, columnspacing=0.7, **LEG, fontsize=6.0)

# (d,e) real nuclei, Cellpose re-segmentation at three resolutions (sequential grey: finer -> coarser)
Rr = pd.read_csv('reseg_bbbc_all.csv'); x = np.linspace(0.45, 1.5, 300)
shades = {1: '#b4b2a9', 2: '#6f6d66', 4: '#1f1e1c'}
labels = {1: 'native', 2: r'$\frac{1}{2}$ resolution', 4: r'$\frac{1}{4}$ resolution'}
for a, k, ttl in [(ax_d, 'sk', '(d) real nuclei, default chain code'), (ax_e, 'ours', '(e) same nuclei, F1 (ours)')]:
    for f in [1, 2, 4]:
        v = Rr[f'FF_{k}_{f}'].values; v = v[(v > 0.3) & (v < 2)]
        a.plot(x, gaussian_kde(v)(x), color=shades[f], lw=1.4, label=labels[f])
        a.axvline(np.median(v), color=shades[f], lw=0.8, ls=':')
    a.axvline(1, color='#999', lw=0.6, zorder=0)
    a.text(1.008, 0.95, 'perfect circle', transform=a.get_xaxis_transform(), fontsize=6.2, color='#666', va='top')
    a.set_xlim(0.45, 1.5); a.set_xlabel('form factor'); a.set_title(ttl, loc='left')
ax_d.set_ylabel('density'); ax_d.set_yticks([]); plt.setp(ax_e.get_yticklabels(), visible=False)
ax_d.legend(loc='upper left', **LEG)
fig.savefig('../paper/figures/fig_main.pdf', bbox_inches='tight'); fig.savefig('fig_main.png', dpi=220, bbox_inches='tight')
