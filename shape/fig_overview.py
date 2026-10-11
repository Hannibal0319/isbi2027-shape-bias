"""Paper 1, overview figure (column width): (a) the same nucleus at three resolutions, default vs RCC form factor;
(b) why: the chain code runs inside the boundary; (c) RCC: chords shorter than the sampling step are missed."""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.path import Path
import cv2
from skimage.measure import find_contours
from shapes import polygon, truth, rasterize
from estimators import p_skimage
from runcorr import crofton

plt.rcParams.update({'font.size': 7, 'font.family': 'serif', 'pdf.fonttype': 42})
# same hues as Fig. 2: default chain code red, marching squares (PyRadiomics) dark yellow, RCC blue
BLUE, YEL, RED, GREYF, INK = '#2a78d6', '#c98500', '#e34948', '#e4e3df', '#1f1e1c'

fig = plt.figure(figsize=(3.4, 2.2))
gs = fig.add_gridspec(2, 6, height_ratios=[1, 1.25], hspace=0.7, wspace=0.3)

# (a) one shape, three resolutions
rng = np.random.default_rng(7)
P0 = polygon('star', 1.0, rng)
A0, L0 = truth(P0); ff_true = 4 * np.pi * A0 / L0 ** 2
for i, r in enumerate([12, 6, 3]):
    a = fig.add_subplot(gs[0, 2 * i:2 * i + 2])
    P = P0 * r; m = rasterize(P, pad=1)
    A = m.sum(); ffd = 4 * np.pi * A / p_skimage(m) ** 2; ffr = 4 * np.pi * A / crofton(m) ** 2
    a.imshow(m, cmap='Greys', vmin=0, vmax=1.6, interpolation='nearest', origin='lower')
    lo = np.floor(P.min(0)) - 1
    a.plot(P[:, 0] - lo[0], P[:, 1] - lo[1], 'k--', lw=0.6)
    a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_visible(False)
    a.set_title(f'radius {r} px', fontsize=6.8, pad=2)
    for dy, lab, v, col in [(-0.08, 'default ', ffd, RED), (-0.30, 'RCC ', ffr, BLUE)]:
        a.text(0.5, dy, lab, transform=a.transAxes, ha='right', va='top', fontsize=6.6, color=INK)
        a.text(0.5, dy, f'{v:.2f}', transform=a.transAxes, ha='left', va='top', fontsize=6.6, color=col,
               fontweight='bold')
    if i == 0:
        a.text(-0.25, 1.25, f'(a) same nucleus, three resolutions (true FF {ff_true:.2f})', transform=a.transAxes,
               fontsize=7.3, fontweight='bold', va='bottom')

# (b) chain code inside the boundary
s = fig.add_subplot(gs[1, 0:3]); R = 3.3; c0 = np.array([0.27, 0.18]); n = 6
yy, xx = np.mgrid[-n:n + 1, -n:n + 1]; m = (((xx - c0[0]) ** 2 + (yy - c0[1]) ** 2) <= R * R).astype(np.uint8)
for y, x in zip(*np.nonzero(m)):
    s.add_patch(Rectangle((x - n - 0.5, y - n - 0.5), 1, 1, fc=GREYF, ec='white', lw=0.4))
t = np.linspace(0, 2 * np.pi, 200)
s.plot(c0[0] + R * np.cos(t), c0[1] + R * np.sin(t), 'k--', lw=0.7, label='true')
cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); q = cs[0][:, 0, :] - n; q = np.vstack([q, q[:1]])
s.plot(q[:, 0], q[:, 1], color=RED, lw=1.2, label='chain code')
ms = find_contours(np.pad(m, 1).astype(float), 0.5)[0]
s.plot(ms[:, 1] - 1 - n, ms[:, 0] - 1 - n, color=YEL, lw=1.2, label='marching sq.')
s.set_aspect('equal'); s.set_anchor('W'); s.set_xlim(-4.2, 4.6); s.set_ylim(-4.2, 4.3); s.axis('off')
s.legend(loc='center left', bbox_to_anchor=(0.98, 0.5), frameon=False, fontsize=5.8, handlelength=1.1, labelspacing=0.3)

# (c) zoom on the bottom of a smooth boundary (radius 3.2); lines 0.45 apart, sampled every h = 2.2 along each line
c = fig.add_subplot(gs[1, 3:6]); Rb, h, dl, t0, off = 3.2, 2.2, 0.45, 0.034, 0.468
tt = np.linspace(np.pi * 1.12, np.pi * 1.88, 200)
c.fill_between(Rb * np.cos(tt), Rb + Rb * np.sin(tt), 2.0, color=GREYF, lw=0)
c.plot(Rb * np.cos(tt), Rb + Rb * np.sin(tt), 'k--', lw=0.7)
xs = off + h * np.arange(-2, 3); xs = xs[np.abs(xs) < 3]
for j in range(4):
    y = t0 + j * dl; hw = np.sqrt(Rb ** 2 - (Rb - y) ** 2); ins = np.abs(xs) <= hw
    c.plot([-3.0, 3.0], [y, y], color='#ccc', lw=0.6, zorder=0)
    c.plot(xs[~ins], [y] * (~ins).sum(), 'o', ms=2.0, color='#bbb')
    c.plot(xs[ins], [y] * ins.sum(), 'o', ms=3.0, color=BLUE, zorder=6)
    k = ins.sum()
    if k <= 1:
        c.plot([-hw, hw], [y, y], color=INK, lw=2.4 if k == 0 else 1.2, solid_capstyle='butt', zorder=5)
        if k == 1:
            c.plot(xs[ins][0], y, 'o', ms=6.5, mfc='none', mec=INK, mew=0.9, zorder=7)
        c.text(3.15, y, 'missed chord' if k == 0 else 'single-sample run', fontsize=5.8, va='center', color=INK)
c.annotate('', (xs[1], -0.32), (xs[2], -0.32), arrowprops=dict(arrowstyle='<->', lw=0.6, color=INK, shrinkA=0, shrinkB=0))
c.text((xs[1] + xs[2]) / 2, -0.38, '$h$', ha='center', va='top', fontsize=6.5)
c.set_xlim(-3.0, 5.6); c.set_ylim(-0.75, 1.85); c.axis('off')
y = s.get_position().y1 + 0.03
fig.text(0.03, y, '(b) default traces inside', fontsize=7.3, fontweight='bold')
fig.text(0.53, y, r'(c) RCC: missed $\approx n_1/6$', fontsize=7.3, fontweight='bold')
fig.savefig('../paper/figures/fig_overview.pdf', bbox_inches='tight'); fig.savefig('fig_overview.png', dpi=250, bbox_inches='tight')
