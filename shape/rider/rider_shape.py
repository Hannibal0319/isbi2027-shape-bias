"""Paper 1 on RIDER coffee-break: tumour sphericity per reconstruction (3 slice thicknesses x 2 kernels, test/retest),
PyRadiomics (marching cubes, IBSI) vs spacing-aware Crofton vs run-corrected Crofton (RCC)."""
import sys, numpy as np, pandas as pd, SimpleITK as sitk, logging
from multiprocessing import Pool
from scipy.ndimage import label
sys.path.insert(0, '../../shape')
from radiomics import shape
from crofton3d import sphericity as sph_crofton
from runcorr import crofton as crofton_rc
logging.getLogger('radiomics').setLevel(logging.ERROR)


def job(row):
    m = sitk.ReadImage(row['seg'])
    a = sitk.GetArrayFromImage(m) > 0
    if a.sum() < 10:
        return None
    lab, n = label(a, structure=np.ones((3, 3, 3)))
    if n > 1:                                   # keep the largest lesion component
        a = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
    zz, yy, xx = np.nonzero(a)
    a = np.pad(a[zz.min():zz.max() + 1, yy.min():yy.max() + 1, xx.min():xx.max() + 1], 2)
    sp = m.GetSpacing()[::-1]                    # (z, y, x)
    img = sitk.GetImageFromArray(np.ones(a.shape, np.float32)); img.SetSpacing(sp[::-1])
    msk = sitk.GetImageFromArray(a.astype(np.uint8)); msk.SetSpacing(sp[::-1])
    f = shape.RadiomicsShape(img, msk)
    V = a.sum() * np.prod(sp)
    rec = dict(row)
    rec.update(nvox=int(a.sum()), vol=V, nslices=int((a.sum((1, 2)) > 0).sum()),
               prad=f.getSphericityFeatureValue(), crofton=sph_crofton(a, sp),
               rcc=(36 * np.pi * V ** 2) ** (1 / 3) / crofton_rc(a, tuple(sp), True))
    return rec


if __name__ == '__main__':
    idx = pd.read_csv('index.csv')
    with Pool(8) as p:
        res = [r for r in p.map(job, idx.to_dict('records')) if r]
    df = pd.DataFrame(res)
    df.to_csv('rider_shape.csv', index=False)
    print(len(df), 'series')
    print(df.groupby(['thickness', 'kernel'])[['prad', 'crofton', 'rcc', 'nslices']].mean().round(3).to_string())
