"""Paper 1, overview figure (column width, two rows of one idea each):
(a) the same nucleus digitized at radius 12 and 3 px: default vs RCC circularity;
(b) RCC: lines sampled every h miss chords shorter than h; on average they are one sixth of the single-sample runs."""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shapes import polygon, truth, rasterize
from estimators import p_skimage
from runcorr import crofton

plt.rcParams.update({'font.size': 7, 'font.family': 'serif', 'pdf.fonttype': 42})
BLUE, RED, GREYF, INK = '#2a78d6', '#e34948', '#e4e3df', '#1f1e1c'

fig = plt.figure(figsize=(3.4, 2.45))
fig.text(0.0, 0.985, '(a) one nucleus, two resolutions', fontsize=7.4, fontweight='bold', va='top')
fig.text(0.0, 0.47, r'(b) RCC: missed chords $\approx\frac{1}{6}$ of single-sample runs', fontsize=7.4,
         fontweight='bold', va='top')

# (a) radius 12 and 3 px, values to the right of each mask
rng = np.random.default_rng(7)
P0 = polygon('star', 1.0, rng)
A0, L0 = truth(P0); ff_true = 4 * np.pi * A0 / L0 ** 2
for i, r in enumerate([12, 3]):
    ax = fig.add_axes([0.0 + 0.5 * i, 0.53, 0.22, 0.37])
    P = P0 * r; m = rasterize(P, pad=1)
    A = m.sum(); ffd = 4 * np.pi * A / p_skimage(m) ** 2; ffr = 4 * np.pi * A / crofton(m) ** 2
    ax.imshow(m, cmap='Greys', vmin=0, vmax=1.6, interpolation='nearest', origin='lower')
    lo = np.floor(P.min(0)) - 1
    ax.plot(P[:, 0] - lo[0], P[:, 1] - lo[1], 'k--', lw=0.6)
    ax.axis('off')
    tx = 0.235 + 0.5 * i
    fig.text(tx, 0.83, f'radius {r} px', fontsize=6.8, va='center', color=INK)
    for y, lab, v, col in [(0.70, 'default', ffd, RED), (0.60, 'RCC', ffr, BLUE)]:
        fig.text(tx, y, lab, fontsize=6.6, va='center', color=INK)
        fig.text(tx + 0.12, y, f'{v:.2f}', fontsize=6.8, va='center', color=col, fontweight='bold')
fig.text(0.5, 0.505, f'true circularity {ff_true:.2f}; the default drifts with size, RCC does not',
         fontsize=6.2, ha='center', color='#555')

# (b) zoom on the bottom of a smooth boundary; lines 0.45 apart, sampled every h = 2.2 along each line
c = fig.add_axes([0.0, 0.0, 1.0, 0.38]); Rb, h, dl, t0, off = 3.2, 2.2, 0.45, 0.034, 0.468
tt = np.linspace(np.pi * 1.12, np.pi * 1.88, 200)
c.fill_between(Rb * np.cos(tt), Rb + Rb * np.sin(tt), 1.75, color=GREYF, lw=0)
c.plot(Rb * np.cos(tt), Rb + Rb * np.sin(tt), 'k--', lw=0.7)
c.text(0, 1.45, 'object', ha='center', fontsize=6, color='#666')
xs = off + h * np.arange(-2, 3); xs = xs[np.abs(xs) < 3]
for j in range(4):
    y = t0 + j * dl; hw = np.sqrt(Rb ** 2 - (Rb - y) ** 2); ins = np.abs(xs) <= hw
    c.plot([-3.0, 3.0], [y, y], color='#ccc', lw=0.6, zorder=0)
    c.plot(xs[~ins], [y] * (~ins).sum(), 'o', ms=2.2, color='#bbb')
    c.plot(xs[ins], [y] * ins.sum(), 'o', ms=3.2, color=BLUE, zorder=6)
    k = ins.sum()
    if k <= 1:
        c.plot([-hw, hw], [y, y], color=INK, lw=2.6 if k == 0 else 1.3, solid_capstyle='butt', zorder=5)
        if k == 1:
            c.plot(xs[ins][0], y, 'o', ms=7, mfc='none', mec=INK, mew=0.9, zorder=7)
        c.annotate('chord missed (no sample)' if k == 0 else 'single-sample run ($n_1$)', (3.0, y), xytext=(3.4, y),
                   fontsize=6.3, va='center', color=INK, arrowprops=dict(arrowstyle='-', color='#bbb', lw=0.5))
c.annotate('', (xs[1], -0.3), (xs[2], -0.3), arrowprops=dict(arrowstyle='<->', lw=0.6, color=INK, shrinkA=0, shrinkB=0))
c.text((xs[1] + xs[2]) / 2, -0.36, 'sampling step $h$', ha='center', va='top', fontsize=6.3)
c.set_xlim(-3.3, 8.6); c.set_ylim(-0.8, 1.75); c.set_aspect('equal'); c.axis('off')
fig.savefig('../paper/figures/fig_overview.pdf', bbox_inches='tight'); fig.savefig('fig_overview.png', dpi=250, bbox_inches='tight')
