"""Paper 1, Figure 1: two-row layout.
Top: (a) traced polygons on a digitized disk, (b) 2D circularity bias vs size, (c) 3D sphericity bias vs slice thickness.
Bottom: (d,e) form factor of real nuclei re-segmented at three resolutions, default estimator vs F1."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import gaussian_kde
import cv2
from skimage.measure import find_contours
from estimators import KAPPA, _pixel_edge_polygon

plt.rcParams.update({'font.size': 7.5, 'font.family': 'serif', 'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
                     'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
                     'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'legend.fontsize': 6.8,
                     'axes.titlesize': 7.8, 'axes.titleweight': 'bold'})
# one colour per method, used in every panel
COL = {'default': '#2a78d6', 'ImageJ': '#1baf7a', 'PyRadiomics': '#e87ba4', 'iso': '#4a3aa7',
       'Crofton': '#7a7974', 'F1': '#eb6834', 'RCC': '#e34948'}

fig = plt.figure(figsize=(7.1, 2.6))
gs = fig.add_gridspec(2, 12, height_ratios=[1, 0.72], hspace=0.72, wspace=3.6)
ax_a = fig.add_subplot(gs[0, 0:2])
ax_b = fig.add_subplot(gs[0, 2:8])
ax_c = fig.add_subplot(gs[0, 8:12])
ax_d = fig.add_subplot(gs[1, 0:6])
ax_e = fig.add_subplot(gs[1, 6:12], sharey=ax_d)

# (a) schematic
s = ax_a; R = 3.3; c0 = np.array([0.27, 0.18]); n = 6
yy, xx = np.mgrid[-n:n + 1, -n:n + 1]; m = (((xx - c0[0]) ** 2 + (yy - c0[1]) ** 2) <= R * R).astype(np.uint8)
for y, x in zip(*np.nonzero(m)):
    s.add_patch(Rectangle((x - n - 0.5, y - n - 0.5), 1, 1, fc='#e4e3df', ec='white', lw=0.4))
t = np.linspace(0, 2 * np.pi, 200)
s.plot(c0[0] + R * np.cos(t), c0[1] + R * np.sin(t), 'k--', lw=0.8, label='true boundary')
cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); q = cs[0][:, 0, :] - n; q = np.vstack([q, q[:1]])
s.plot(q[:, 0], q[:, 1], color=COL['default'], lw=1.2, label='chain code')
ms = find_contours(np.pad(m, 1).astype(float), 0.5)[0]
s.plot(ms[:, 1] - 1 - n, ms[:, 0] - 1 - n, color=COL['PyRadiomics'], lw=1.2, label='marching sq.')
s.set_aspect('equal'); s.set_xlim(-4.6, 4.8); s.set_ylim(-4.6, 4.6); s.axis('off')
s.set_title('(a) outlines', loc='left', x=-0.42)

# (b) 2D circularity bias vs size
df = pd.read_csv('exp1_synth_rc.csv'); b = ax_b
series = [('P_skimage/CellProfiler', 'skimage / CellProfiler / OpenCV', 'default', 'o'),
          ('P_ImageJ', 'ImageJ', 'ImageJ', '^'),
          ('P_PyRadiomics', 'PyRadiomics (IBSI)', 'PyRadiomics', 'v'),
          ('P_Crofton (skimage)', 'Crofton', 'Crofton', 'x'),
          ('P_corrected chain (ours)', 'F1, post hoc (ours)', 'F1', 's'),
          ('P_rc', 'RCC (ours)', 'RCC', 'P')]
for k, lab, ck, mk in series:
    est = 4 * np.pi * df.Ad / df[k] ** 2
    g = (est - df.ff_true).groupby(df.r).mean()
    b.plot(g.index, g.values, marker=mk, ms=3, lw=1.1, color=COL[ck], label=lab)
th = (df.ff_true * ((df.L / (KAPPA * (df.L - 2 * np.sqrt(2)))) ** 2 - 1)).groupby(df.r).mean()
r = th.index[th.index >= 3]
b.plot(r, th[r], color='k', ls=':', lw=1.1, label='model, Eq. (2)', zorder=5)
b.axhline(0, color='#bbb', lw=0.6, zorder=0)
b.set_xscale('log'); b.set_xticks([2, 4, 8, 16, 32]); b.set_xticklabels([2, 4, 8, 16, 32])
b.set_ylim(-0.12, 0.6); b.set_xlabel('object radius (pixels)'); b.set_ylabel('circularity bias')
b.set_title('(b) 2D circularity error vs. object size', loc='left')

# (c) 3D sphericity bias vs slice thickness
d3 = pd.read_csv('exp3d_synth.csv'); d3 = d3[d3.R >= 5]; c = ax_c
for col, lab, ck, mk, ls in [('prad', 'PyRadiomics', 'PyRadiomics', 'v', '-'),
                             ('prad_iso', 'PyRadiomics, resampled', 'iso', 'D', '--'),
                             ('crofton', 'Crofton', 'Crofton', 'x', '-'),
                             ('rcc', 'RCC (ours)', 'RCC', 'P', '-')]:
    g = (d3[col] - d3.true).groupby(d3.slice)
    c.plot(g.mean().index, g.mean().values, marker=mk, ms=3, lw=1.1, ls=ls, color=COL[ck], label=lab)
    c.fill_between(g.mean().index, g.quantile(0.1), g.quantile(0.9), color=COL[ck], alpha=0.13, lw=0)
c.axhline(0, color='#bbb', lw=0.6, zorder=0)
c.set_xlabel('slice thickness (mm)'); c.set_ylabel('sphericity bias'); c.set_ylim(-0.24, 0.05); c.set_yticks([0, -0.1, -0.2]); c.yaxis.labelpad = 1
c.set_title('(c) 3D sphericity error', loc='left')

# (d,e) real nuclei, Cellpose re-segmentation at three resolutions
Rr = pd.read_csv('reseg_bbbc_all.csv'); x = np.linspace(0.45, 1.5, 300)
shades = {1: '#9ec5f4', 2: '#2a78d6', 4: '#0b3e7e'}
labels = {1: 'native', 2: r'$\frac{1}{2}$ resolution', 4: r'$\frac{1}{4}$ resolution'}
for a, k, ttl in [(ax_d, 'sk', '(d) real nuclei: default estimator'),
                  (ax_e, 'ours', '(e) same nuclei: F1 (ours)')]:
    for f in [1, 2, 4]:
        v = Rr[f'FF_{k}_{f}'].values; v = v[(v > 0.3) & (v < 2)]
        a.plot(x, gaussian_kde(v)(x), color=shades[f], lw=1.3, label=labels[f])
        a.axvline(np.median(v), color=shades[f], lw=0.8, ls=':')
    a.axvline(1, color='#999', lw=0.6, zorder=0)
    a.text(1.005, 0.97, 'perfect\ncircle', transform=a.get_xaxis_transform(), fontsize=6, color='#777', va='top')
    a.set_xlim(0.45, 1.5); a.set_xlabel('form factor (circularity)'); a.set_title(ttl, loc='left')
ax_d.set_ylabel('density'); ax_d.set_yticks([]); plt.setp(ax_e.get_yticklabels(), visible=False)
ax_d.legend(loc='upper left', frameon=False, handlelength=1.6, labelspacing=0.2, borderaxespad=0)
hb, lb = ax_b.get_legend_handles_labels(); hc, lc = ax_c.get_legend_handles_labels()
H = dict(zip(lb, hb)); H.update({l: h for h, l in zip(hc, lc) if l not in ('PyRadiomics', 'Crofton', 'RCC (ours)')})
order = ['skimage / CellProfiler / OpenCV', 'ImageJ', 'PyRadiomics (IBSI)', 'PyRadiomics, resampled', 'Crofton', 'F1, post hoc (ours)', 'RCC (ours)', 'model, Eq. (2)']
fig.legend([H[k] for k in order], order, loc='upper center', bbox_to_anchor=(0.5, 1.09), ncol=4, frameon=False, handlelength=1.8, columnspacing=1.4, labelspacing=0.3)
fig.savefig('../paper/figures/fig_main.pdf', bbox_inches='tight'); fig.savefig('fig_main.png', dpi=220, bbox_inches='tight')
