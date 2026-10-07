import numpy as np, pandas as pd, sys
from multiprocessing import Pool
from runcorr import crofton
def work(args):
    f,idx=args; d=np.load(f); return [crofton(d[f'm{i}']) for i in idx]
if __name__=='__main__':
    for name in ['real_pannuke_2656','real_bbbc_670']:
        df=pd.read_csv(name+'.csv'); f=name+'_crops.npz'; n=len(df)
        chunks=[(f,list(range(i,min(i+2000,n)))) for i in range(0,n,2000)]
        with Pool(8) as p: res=p.map(work,chunks)
        P=np.concatenate([np.array(r) for r in res]); df['P_rc']=P*df.scale
        df.to_csv(name+'.csv',index=False); print(name,len(df))
