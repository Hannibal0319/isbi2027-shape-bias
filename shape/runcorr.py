"""Crofton estimators with run-length (missed-chord) correction, 2D and 3D, arbitrary spacing."""
import numpy as np, itertools
from functools import lru_cache

def _shift_pair(m, d):
    a = m[tuple(slice(max(-k, 0), m.shape[i] - max(k, 0)) for i, k in enumerate(d))]
    b = m[tuple(slice(max(k, 0), m.shape[i] - max(-k, 0)) for i, k in enumerate(d))]
    return a, b

def run_counts(m, d):
    """N = number of runs (object->background exits) along d; n1 = runs of exactly one sample."""
    m = np.pad(np.asarray(m, bool), 2)
    a, b = _shift_pair(m, d)                  # a = p, b = p+d
    N = np.count_nonzero(a & ~b)
    d2 = tuple(2 * k for k in d)
    # single-sample run: p in, p+d out, p-d out  -> compare p with both neighbours
    pad = np.pad(m, [(abs(k), abs(k)) for k in d])
    sl = lambda off: pad[tuple(slice(abs(k) + o, abs(k) + o + n) for k, o, n in zip(d, off, m.shape))]
    prev = sl(tuple(-k for k in d)); nxt = sl(d)
    n1 = np.count_nonzero(m & ~prev & ~nxt)
    return N, n1

def dirs(ndim):
    return [d for d in itertools.product([-1, 0, 1], repeat=ndim) if d > (0,) * ndim]

@lru_cache(maxsize=64)
def weights(spacing, n=2_000_000, seed=0):
    D = np.array(dirs(len(spacing)), float) * np.array(spacing); D /= np.linalg.norm(D, axis=1, keepdims=True)
    u = np.random.default_rng(seed).normal(size=(n, len(spacing))); u /= np.linalg.norm(u, axis=1, keepdims=True)
    return np.bincount(np.argmax(np.abs(u @ D.T), axis=1), minlength=len(D)) / n

def run_counts2(m, d):
    """N (runs), n1 (one-sample runs), n2 (two-sample runs) along lattice direction d."""
    m = np.pad(np.asarray(m, bool), 4)
    pad = np.pad(m, [(4 * abs(k), 4 * abs(k)) for k in d])
    sl = lambda j: pad[tuple(slice(4 * abs(k) + j * k, 4 * abs(k) + j * k + n) for k, n in zip(d, m.shape))]
    start = m & ~sl(-1)
    return (np.count_nonzero(start), np.count_nonzero(start & ~sl(1)),
            np.count_nonzero(start & sl(1) & ~sl(2)))

def crofton(m, spacing=None, corrected=True, order=1):
    """Perimeter (2D) or surface area (3D). corrected=True adds the missed chords per direction:
    n1/6 (order 1, linear chord-length density) or (19 n1 - 4 n2)/66 (order 2, quadratic density)."""
    m = np.asarray(m, bool); nd = m.ndim
    sp = np.ones(nd) if spacing is None else np.array(spacing, float)
    w = weights(tuple(float(s) for s in sp)); v = np.prod(sp); tot = 0.0
    for wd, d in zip(w, dirs(nd)):
        if order == 2:
            N, n1, n2 = run_counts2(m, d)
            if corrected: N = N + max(0.0, (19 * n1 - 4 * n2) / 66.0)
        else:
            N, n1 = run_counts(m, d)
            if corrected: N = N + n1 / 6.0
        tot += wd * N * v / np.linalg.norm(np.array(d) * sp)
    return (np.pi if nd == 2 else 4.0) * tot
