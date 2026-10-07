"""Per-nucleus shape features at native and re-digitized (1/2, 1/4) resolution for PanNuke / BBBC038."""
import numpy as np, pandas as pd, glob, sys
from multiprocessing import Pool
from skimage.measure import regionprops, label
from estimators import PERIMETERS
SCALES=[1,2,4]
def down(m,f):
    if f==1: return m
    H,W=m.shape; return (m.reshape(H//f,f,W//f,f).mean((1,3))>=0.5).astype(np.uint8)
def nucleus_rows(m, meta):
    """m: full-size binary mask of one nucleus (H,W multiple of 4). Returns list of row dicts or []."""
    ys,xs=np.nonzero(m)
    if len(ys)<20 or ys.min()==0 or xs.min()==0 or ys.max()==m.shape[0]-1 or xs.max()==m.shape[1]-1: return []
    y0=max(ys.min()//4*4-8,0); x0=max(xs.min()//4*4-8,0); y1=min((ys.max()//4+3)*4,m.shape[0]); x1=min((xs.max()//4+3)*4,m.shape[1])
    if (y1-y0)%4 or (x1-x0)%4: return []
    crop=m[y0:y1,x0:x1].astype(np.uint8); out=[]
    for f in SCALES:
        c=down(crop,f)
        if c.sum()<3: return []
        lab=label(c,connectivity=2)
        if lab.max()!=1: return []
        c=np.pad(c,2); p=regionprops(c)[0]
        rec=dict(meta, scale=f, A=p.area*f*f, solidity=p.solidity, ecc=p.eccentricity, extent=p.extent,
                 major=p.axis_major_length*f, minor=p.axis_minor_length*f, euler=p.euler_number)
        for k,fn in PERIMETERS.items(): rec['P_'+k]=fn(c)*f
        rec['_crop']=c
        out.append(rec)
    return out
def pannuke_patch(i):
    M=np.load(PN,mmap_mode='r'); x=np.array(M[i]); rows=[]
    for ch in range(5):
        for iid in np.unique(x[...,ch]):
            if iid==0: continue
            rows+=nucleus_rows((x[...,ch]==iid).astype(np.uint8), dict(src='pannuke',img=i,cls=ch,iid=int(iid)))
    return rows
def bbbc_img(d):
    from skimage.io import imread
    rows=[]
    for j,f in enumerate(sorted(glob.glob(d+'/masks/*.png'))):
        m=(imread(f)>0).astype(np.uint8); H,W=m.shape; m=m[:H//4*4,:W//4*4]
        rows+=nucleus_rows(m, dict(src='bbbc038',img=d[-64:],cls=-1,iid=j))
    return rows
PN='../data/pannuke/fold1/Fold 1/masks/fold1/masks.npy'
if __name__=='__main__':
    which,n=sys.argv[1],int(sys.argv[2])
    with Pool(11) as pool:
        if which=='pannuke':
            N=np.load(PN,mmap_mode='r').shape[0]; idx=np.random.default_rng(0).permutation(N)[:n]
            R=pool.map(pannuke_patch, idx, chunksize=4)
        else:
            ds=sorted(glob.glob('../data/bbbc038/train/*'))[:n]; R=pool.map(bbbc_img, ds, chunksize=2)
    rows=[r for rr in R for r in rr]
    np.savez_compressed(f'real_{which}_{n}_crops.npz',**{f'm{i}':r.pop('_crop') for i,r in enumerate(rows)})
    df=pd.DataFrame(rows); df.to_csv(f'real_{which}_{n}.csv',index=False); print(which,len(df)//3,'nuclei')
