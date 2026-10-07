"""Assign test/retest visits: same-day repeats share StudyDate (and often the study), so cluster each patient's
acquisition times; a new visit starts where consecutive acquisition times differ by more than 120 s."""
import glob, numpy as np, pandas as pd, pydicom

i = pd.read_csv('index.csv')
acq = []
for s in i.series:
    d = pydicom.dcmread(glob.glob(f'ct/{s}/*.dcm')[0], stop_before_pixels=True)
    acq.append(str(d.StudyDate) + str(d.get('AcquisitionTime', '')).split('.')[0].zfill(6))
i['acq'] = acq


def secs(a):
    return int(a[:8]) * 86400 + int(a[8:10]) * 3600 + int(a[10:12]) * 60 + int(a[12:14])


i['t'] = i.acq.map(secs)
visits = pd.Series(0, index=i.index)
for p, g in i.groupby('patient'):
    u = np.sort(g.t.unique())
    v = np.cumsum(np.r_[1, np.diff(u) > 120])
    visits[g.index] = g.t.map(dict(zip(u, v)))
i['visit'] = visits.astype(int)
i.to_csv('index.csv', index=False)
print('visits per patient', i.groupby('patient').visit.max().value_counts().to_dict())
