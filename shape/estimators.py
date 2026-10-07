"""Perimeter estimators used (or emulated) by common open-source bioimage / radiomics tools.

All functions take a 2D binary mask containing one object (uint8/bool) and return a perimeter in pixels.
"""
import numpy as np
import cv2
from skimage.measure import perimeter as _sk_perimeter, perimeter_crofton as _sk_crofton

KAPPA = 8 * (np.sqrt(2) - 1) / np.pi  # orientation-averaged length factor of 8-chain codes (=1.0548)


def p_skimage(m):
    """skimage.measure.regionprops(...).perimeter  (also CellProfiler FormFactor, HistomicsTK, MATLAB<R2023a)."""
    return _sk_perimeter(m, 4)


def p_opencv(m):
    """cv2.arcLength on the outer contour from cv2.findContours (CHAIN_APPROX_NONE)."""
    cs, _ = cv2.findContours(np.ascontiguousarray(m, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea)
    if len(c) == 1:
        return 0.0
    return cv2.arcLength(c, True)


def p_crofton(m):
    """skimage.measure.perimeter_crofton(m, 4) (Cauchy-Crofton, 4 directions)."""
    return _sk_crofton(m, 4)


def p_corrected(m):
    """Proposed post-hoc correction of the chain-code perimeter: P/kappa + pi (orientation + Steiner offset)."""
    return p_skimage(m) / KAPPA + np.pi


def correct_perimeter(P):
    """Same correction applied to an existing perimeter column (e.g. a CellProfiler table)."""
    return np.asarray(P) / KAPPA + np.pi


def _pixel_edge_polygon(m):
    """Outer boundary of the union of pixel squares as a closed lattice polygon (vertices at corners),
    traced counter-clockwise; collinear vertices removed. Diagonal-only contacts are resolved by turning left
    (8-connected foreground)."""
    m = np.pad(np.asarray(m, bool), 1)
    H, W = m.shape
    nxt = {}
    # directed edges with the foreground on the left; coordinates (x, y) of pixel corners, y down
    ys, xs = np.nonzero(m)
    for y, x in zip(ys, xs):
        if not m[y - 1, x]:
            nxt.setdefault((x + 1, y), []).append((x, y))          # top edge, going left
        if not m[y + 1, x]:
            nxt.setdefault((x, y + 1), []).append((x + 1, y + 1))  # bottom edge, going right
        if not m[y, x - 1]:
            nxt.setdefault((x, y), []).append((x, y + 1))          # left edge, going down
        if not m[y, x + 1]:
            nxt.setdefault((x + 1, y + 1), []).append((x + 1, y))  # right edge, going up
    # start at the top-left-most corner of the object, which lies on the outer boundary
    y0, x0 = ys[0], xs[ys == ys[0]].min()
    start = (x0 + 1, y0)
    pts = [start]
    prev_dir = None
    cur = start
    for _ in range(4 * m.size):
        cands = nxt[cur]
        if len(cands) == 1 or prev_dir is None:
            n = cands[0]
        else:  # choose the candidate making the sharpest left turn w.r.t. previous direction
            def turn(c):
                d = (c[0] - cur[0], c[1] - cur[1])
                return prev_dir[0] * d[1] - prev_dir[1] * d[0]
            n = max(cands, key=turn)
        prev_dir = (n[0] - cur[0], n[1] - cur[1])
        cur = n
        if cur == start:
            break
        pts.append(cur)
    P = np.array(pts)
    # remove collinear vertices
    d1 = P - np.roll(P, 1, 0)
    d2 = np.roll(P, -1, 0) - P
    keep = (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) != 0
    return P[keep]


def p_pixel_edges(m):
    """Length of the pixel-edge polygon (QuPath / shapely / rasterio polygonised masks)."""
    P = _pixel_edge_polygon(m)
    return float(np.sum(np.abs(np.roll(P, -1, 0) - P)))


def n_exposed_edges(m):
    m = np.pad(np.asarray(m, bool), 1).astype(np.int8)
    return int(np.abs(np.diff(m, axis=0)).sum() + np.abs(np.diff(m, axis=1)).sum())


def p_imagej(m):
    """ImageJ PolygonRoi.getTracedPerimeter on the wand-traced pixel-edge polygon."""
    P = _pixel_edge_polygon(m)
    xp, yp = P[:, 0], P[:, 1]
    n = len(xp)
    if n < 4:
        return 0.0
    sumdx = sumdy = ncorners = 0
    dx1, dy1 = xp[0] - xp[n - 1], yp[0] - yp[n - 1]
    side1 = abs(dx1) + abs(dy1)
    corner = False
    for i in range(n):
        j = (i + 1) % n
        dx2, dy2 = xp[j] - xp[i], yp[j] - yp[i]
        sumdx += abs(dx1)
        sumdy += abs(dy1)
        side2 = abs(dx2) + abs(dy2)
        if side1 > 1 or not corner:
            corner = True
            ncorners += 1
        else:
            corner = False
        dx1, dy1, side1 = dx2, dy2, side2
    return sumdx + sumdy - ncorners * (2 - np.sqrt(2))


def ff(A, P):
    return 4 * np.pi * A / P ** 2


def ff_matlab2023(A, P_chain):
    """MATLAB regionprops Circularity since R2023a (bias-corrected)."""
    r = P_chain / (2 * np.pi) + 0.5
    return 4 * np.pi * A / P_chain ** 2 * (1 - 0.5 / r) ** 2


PERIMETERS = {
    'skimage/CellProfiler': p_skimage,
    'OpenCV': p_opencv,
    'ImageJ': p_imagej,
    'pixel-edge (QuPath)': p_pixel_edges,
    'Crofton (skimage)': p_crofton,
    'corrected chain (ours)': p_corrected,
}
