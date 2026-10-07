"""Paper 2 on RIDER coffee-break.
(a) soft tissue with internal reference, per real reconstruction (kernel x slice thickness) -> natural correlation regimes;
(b) tumour texture per reconstruction: plug-in vs block jackknife (gated by N/xi^3)."""
import sys, numpy as np, pandas as pd, SimpleITK as sitk
from multiprocessing import Pool
from scipy.ndimage import median_filter, distance_transform_edt, label
sys.path.insert(0, '../../texture')
from texfeat import features
from corrections import evaluate
from real_regions import ball, corr_length

RADII = [3, 4, 5, 6, 8]


def soft_tissue(row):
    rng = np.random.default_rng(abs(hash(row['series'])) % 2**31)
    img = sitk.GetArrayFromImage(sitk.ReadImage(row['ct'])).astype(np.float32)
    soft = median_filter(((img > -50) & (img < 150)).astype(np.uint8), 3).astype(bool)
    dist = distance_transform_edt(soft)
    out = []
    for RB in (14, 10):
        cand = np.argwhere(dist > RB + 1)
        if len(cand):
            break
    else:
        return out
    for t in range(3):
        c = cand[rng.integers(len(cand))]
        sub = img[tuple(slice(ci - RB - 2, ci + RB + 3) for ci in c)]
        if sub.shape != (2 * RB + 5,) * 3:
            continue
        B = np.zeros(sub.shape, bool); B[2:-2, 2:-2, 2:-2] = ball(RB)
        ref = features(sub, B)
        xi = corr_length(sub[2:-2, 2:-2, 2:-2][3:-3, 3:-3, 3:-3])
        for r in [x for x in RADII if x <= RB - 2]:
            for s in range(3):
                off = rng.integers(-(RB - r), RB - r + 1, 3)
                if np.sum(off ** 2) > (RB - r) ** 2:
                    continue
                m = np.zeros(sub.shape, bool)
                cc = np.array(sub.shape) // 2 + off
                m[tuple(slice(ci - r, ci + r + 1) for ci in cc)] = ball(r)
                keys, est = evaluate(features, sub, m, 2, halves=False)
                roi = f"{row['series']}_{t}_{r}_{s}"
                for name, v in est.items():
                    for kk, vv in zip(keys, v):
                        out.append(dict(part='soft', roi=roi, patient=row['patient'], visit=row['visit'],
                                        thickness=row['thickness'], kernel=row['kernel'], RB=RB, xi=xi, r=r,
                                        N=int(m.sum()), method=name, feature=kk, value=vv, ref=ref[kk]))
    return out


def tumour(row):
    img = sitk.GetArrayFromImage(sitk.ReadImage(row['ct'])).astype(np.float32)
    a = sitk.GetArrayFromImage(sitk.ReadImage(row['seg'])) > 0
    if a.sum() < 30:
        return []
    lab, n = label(a, structure=np.ones((3, 3, 3)))
    if n > 1:
        a = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
    zz, yy, xx = np.nonzero(a)
    sl = tuple(slice(max(v.min() - 3, 0), v.max() + 4) for v in (zz, yy, xx))
    sub, m = img[sl], a[sl]
    xi = corr_length(np.where(m, sub, sub[m].mean())) if min(sub.shape) > 8 else np.nan
    keys, est = evaluate(features, sub, m, 2, halves=False)
    out = []
    for name, v in est.items():
        for kk, vv in zip(keys, v):
            out.append(dict(part='tumour', roi=row['series'], patient=row['patient'], visit=row['visit'],
                            thickness=row['thickness'], kernel=row['kernel'], xi=xi, N=int(m.sum()),
                            method=name, feature=kk, value=vv))
    return out


def job(row):
    return soft_tissue(row) + tumour(row)


if __name__ == '__main__':
    idx = pd.read_csv('index.csv')
    with Pool(8) as p:
        R = p.map(job, idx.to_dict('records'))
    df = pd.DataFrame([x for rr in R for x in rr])
    df.to_csv('rider_texture.csv', index=False)
    s = df[df.part == 'soft']
    print('soft ROIs', s.roi.nunique(), 'tumours', df[df.part == 'tumour'].roi.nunique())
    print(s.drop_duplicates('roi').groupby(['kernel', 'thickness']).xi.median().round(2).to_string())
