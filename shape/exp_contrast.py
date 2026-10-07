"""Effect of the perimeter estimator on biological contrasts (PanNuke, annotated nucleus types)."""
import numpy as np, pandas as pd
from estimators import KAPPA
df=pd.read_csv('real_pannuke_2656.csv'); df['P_PyRadiomics']=np.load('real_pannuke_2656_prad.npy')[:,0]*df.scale
P=df['P_skimage/CellProfiler']/df.scale
E={'skimage / CellProfiler':4*np.pi*df.A/df['P_skimage/CellProfiler']**2,
   'OpenCV':4*np.pi*df.A/df['P_OpenCV']**2,'ImageJ':4*np.pi*df.A/df['P_ImageJ']**2,
   'PyRadiomics':4*np.pi*df.A/df['P_PyRadiomics']**2,'Crofton':4*np.pi*df.A/df['P_Crofton (skimage)']**2,
   'F1 (ours)':4*np.pi*df.A/((P/KAPPA+np.pi)*df.scale)**2,'RCC (ours)':4*np.pi*df.A/df['P_rc']**2}
df=df[df.groupby(['img','cls','iid']).scale.transform('count')==3]
def d(a,b): return (a.mean()-b.mean())/np.sqrt((a.var()+b.var())/2)
rng=np.random.default_rng(0)
rows={}
for n,v in E.items():
    v=v.loc[df.index]; g=lambda c,s: v[(df.cls==c)&(df.scale==s)]
    r={'inf40':d(g(1,1),g(0,1)),'inf10':d(g(1,4),g(0,4)),'con40':d(g(2,1),g(0,1)),'con10':d(g(2,4),g(0,4)),'mixed':d(g(1,2),g(0,1))}
    rows[n]=r
T=pd.DataFrame(rows).T; print(T.round(2).to_string())
# bootstrap s.e. of d for default inf40
v=E['skimage / CellProfiler'].loc[df.index]; a=v[(df.cls==1)&(df.scale==1)].values; b=v[(df.cls==0)&(df.scale==1)].values
bs=[d(pd.Series(rng.choice(a,len(a))),pd.Series(rng.choice(b,len(b)))) for _ in range(200)]; print('se',np.std(bs), len(a), len(b))
