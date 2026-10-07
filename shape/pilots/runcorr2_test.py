import numpy as np, pandas as pd
from runcorr import dirs, weights
def runs_hist(m,d,K=2):
    m=np.pad(np.asarray(m,bool),4)
    pad=np.pad(m,[(4*abs(k),4*abs(k)) for k in d])
    sl=lambda j: pad[tuple(slice(4*abs(k)+j*k,4*abs(k)+j*k+n) for k,n in zip(d,m.shape))]
    start=m & ~sl(-1)                       # run starts
    N=np.count_nonzero(start)
    n1=np.count_nonzero(start & ~sl(1))
    n2=np.count_nonzero(start & sl(1) & ~sl(2))
    return N,n1,n2
def cro(m,order):
    sp=np.ones(m.ndim); w=weights(tuple(sp)); tot=0
    for wd,d in zip(w,dirs(m.ndim)):
        N,n1,n2=runs_hist(m,d)
        if order==1: N=N+n1/6
        if order==2: N=N+max(0,(19*n1-4*n2)/66)
        tot+=wd*N/np.linalg.norm(d)
    return np.pi*tot
df=pd.read_csv('exp1_synth_rc.csv'); M=np.load('exp1_masks.npz')
for o in [0,1,2]: df[f'P{o}']=[cro(M[f'm{i}'],o) for i in range(len(df))]
for o in [0,1,2]:
    e=4*np.pi*df.Ad/df[f'P{o}']**2-df.ff_true
    print('order',o); print(e.groupby([df.kind,df.r]).mean().unstack('r')[[2,3,4,8,16]].round(3).to_string()); print('sd r=2,4:',e[df.r==2].std().round(3),e[df.r==4].std().round(3))
