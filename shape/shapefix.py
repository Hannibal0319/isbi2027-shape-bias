"""shapefix: drop-in corrections for resolution-dependent shape descriptors.

    from shapefix import perimeter_rcc, surface_rcc, sphericity_rcc, fix_cellprofiler
    P = perimeter_rcc(mask)                       # 2D, run-corrected Crofton perimeter (pixels or physical units)
    S = surface_rcc(mask3d, spacing=(5, .7, .7))  # 3D surface area on the native (anisotropic) grid
    df = fix_cellprofiler(df)                      # repair an existing CellProfiler per-object table (F1)
"""
import numpy as np
from runcorr import crofton as _rcc

KAPPA = 8 * (np.sqrt(2) - 1) / np.pi   # chain-code length factor


def perimeter_rcc(mask, spacing=None):
    """Run-corrected Crofton perimeter of a 2D binary mask."""
    return _rcc(np.asarray(mask, bool), spacing, corrected=True)


def surface_rcc(mask, spacing=(1.0, 1.0, 1.0)):
    """Run-corrected Crofton surface area of a 3D binary mask with voxel spacing (z, y, x)."""
    return _rcc(np.asarray(mask, bool), tuple(float(s) for s in spacing), corrected=True)


def sphericity_rcc(mask, spacing=(1.0, 1.0, 1.0)):
    m = np.asarray(mask, bool)
    V = m.sum() * np.prod(spacing)
    return (36 * np.pi * V ** 2) ** (1 / 3) / surface_rcc(m, spacing)


def correct_chain_perimeter(P, pixel_size=1.0):
    """F1: P_chain / kappa + pi, for perimeters measured by scikit-image regionprops / CellProfiler / OpenCV.
    pixel_size converts the Steiner offset (pi pixels) to the units of P."""
    return np.asarray(P, float) / KAPPA + np.pi * pixel_size


def fix_cellprofiler(df, prefix='AreaShape_', pixel_size=1.0):
    """Repair Perimeter, FormFactor and Compactness columns of a CellProfiler object table in place (copy returned).
    Works for any object set (Nuclei_, Cells_, ...) when prefix is e.g. 'Nuclei_AreaShape_'."""
    out = df.copy()
    A, P = out[prefix + 'Area'].astype(float), out[prefix + 'Perimeter'].astype(float)
    Pc = correct_chain_perimeter(P, pixel_size)
    out[prefix + 'Perimeter'] = Pc
    if prefix + 'FormFactor' in out:
        out[prefix + 'FormFactor'] = 4 * np.pi * A / Pc ** 2
    if prefix + 'Compactness' in out:
        out[prefix + 'Compactness'] = Pc ** 2 / (4 * np.pi * A)
    return out
