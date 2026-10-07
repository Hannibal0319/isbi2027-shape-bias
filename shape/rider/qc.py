"""QC: actual slice spacing and uniformity per CT series (from ImagePositionPatient), merged into index.csv."""
import glob, os, numpy as np, pandas as pd, pydicom

idx = pd.read_csv('index.csv')
rows = []
for s in idx.series:
    pos = []
    for f in glob.glob(f'ct/{s}/*.dcm'):
        d = pydicom.dcmread(f, stop_before_pixels=True, specific_tags=['ImagePositionPatient'])
        pos.append(float(d.ImagePositionPatient[2]))
    z = np.sort(np.array(pos))
    dz = np.diff(z)
    med = float(np.median(dz)) if len(dz) else np.nan
    rows.append(dict(series=s, nslices_dcm=len(z), dz_median=med, dz_max_dev=float(np.max(np.abs(dz - med))) if len(dz) else np.nan,
                     dup=int((dz < 1e-3).sum())))
q = pd.DataFrame(rows)
idx = idx.drop(columns=[c for c in q.columns if c in idx.columns and c != 'series']).merge(q, on='series')
idx['uniform'] = (idx.dz_max_dev < 0.05 * idx.dz_median) & (idx.dup == 0)
idx.to_csv('index.csv', index=False)
print(len(idx), 'series; uniform', idx.uniform.sum())
print(idx.groupby(['thickness', 'kernel']).agg(n=('series', 'size'), uniform=('uniform', 'sum'), dz=('dz_median', 'median')).to_string())
