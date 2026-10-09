"""Direct check of Eq. (5): expected number of missed chords vs one sixth of the single-sample runs.
Random ellipses, horizontal lattice lines (unit line spacing) sampled every h, random offsets.
Chord on line y: interval of x where the line is inside the ellipse; it is missed if no sample falls in it,
and is a single-sample run if exactly one does."""
import numpy as np

rng = np.random.default_rng(0)
NIT = {1: 60000, 2: 60000, 4: 60000, 8: 60000, 16: 60000}


def chords(a, b, th, cy, ys):
    # ellipse x'^2/a^2 + y'^2/b^2 <= 1 rotated by th; intersect with horizontal lines y = ys - cy
    c, s = np.cos(th), np.sin(th)
    y = ys - cy
    A = c * c / a ** 2 + s * s / b ** 2
    B = 2 * y * c * s * (1 / a ** 2 - 1 / b ** 2)
    C = y * y * (s * s / a ** 2 + c * c / b ** 2) - 1
    disc = B * B - 4 * A * C
    ok = disc > 0
    r = np.sqrt(disc[ok])
    return (-B[ok] - r) / (2 * A), (-B[ok] + r) / (2 * A)


print('radius  h/r   missed   n1/6   ratio missed/n1')
for R in [1, 2, 4, 8, 16]:
    for h in [1.0, 2.0, 5.0]:
        if h > 2.5 * R:
            continue
        miss = n1 = 0
        for _ in range(NIT[R]):
            q = rng.uniform(0.5, 1.0)                 # axis ratio
            a, b = R / np.sqrt(q), R * np.sqrt(q)
            th = rng.uniform(0, np.pi)
            ys = np.arange(-3 * R - 3, 3 * R + 4) + rng.uniform()
            x0, x1 = chords(a, b, th, 0.0, ys)
            off = rng.uniform(0, h)
            k = np.floor((x1 - off) / h) - np.ceil((x0 - off) / h) + 1   # samples inside each chord
            miss += np.sum(k <= 0); n1 += np.sum(k == 1)
        se = np.sqrt(miss) / n1; print(f"{R:5d}  {h / R:4.2f}  {miss:7d}  {n1 / 6:7.0f}   {miss / n1:.3f} +- {se:.3f}")
