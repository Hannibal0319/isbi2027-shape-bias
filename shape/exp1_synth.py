import numpy as np, pandas as pd, pickle
from shapes import polygon, truth, rasterize
from estimators import PERIMETERS, ff, ff_matlab2023, p_skimage
from skimage.measure import label, regionprops
rng=np.random.default_rng(42)
kinds=['ell1','ell0.6','ell0.35','fourier','irregular']; radii=[2,3,4,6,8,12,16,24,32,48]; N=100
rows=[]; masks=[]
for kind in kinds:
  for r in radii:
    for i in range(N):
      P=polygon(kind,r,rng); A,L=truth(P); m=rasterize(P)
      lab=label(m,connectivity=2)
      if lab.max()!=1: continue   # skip shapes fragmented by digitization
      Ad=m.sum(); rec=dict(kind=kind,r=r,i=i,A=A,L=L,ff_true=ff(A,L),Ad=Ad)
      for k,f in PERIMETERS.items(): rec['P_'+k]=f(m)
      rec['ff_matlab2023']=ff_matlab2023(Ad,rec['P_skimage/CellProfiler'])
      rows.append(rec); masks.append(m)
df=pd.DataFrame(rows); df.to_csv('exp1_synth.csv',index=False)
pickle.dump(masks,open('exp1_masks.pkl','wb'))
print(len(df))
import os
if os.path.exists('exp1_prad.npy'):  # PyRadiomics results (prad_worker.py, run in .venv39) on the same masks
    pr=np.load('exp1_prad.npy')
    if len(pr)==len(df): df['P_PyRadiomics']=pr[:,0]; df.to_csv('exp1_synth.csv',index=False)
