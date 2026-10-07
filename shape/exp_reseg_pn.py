import numpy as np, sys, pandas as pd
from skimage.transform import downscale_local_mean
from skimage.measure import regionprops
from skimage.color import rgb2gray
from cellpose import models
sys.path.insert(0,'.')
from estimators import p_skimage, p_crofton, p_corrected, p_opencv, p_imagej
from runcorr import crofton as crofton_rc
N=int(sys.argv[1]); out=sys.argv[2]
IM=np.load('../data/pannuke/fold1/Fold 1/images/fold1/images.npy',mmap_mode='r')
model=models.Cellpose(gpu=True,model_type='nuclei')
idx=np.random.default_rng(0).permutation(IM.shape[0])[:N]
imgs=[255-rgb2gray(np.array(IM[i]).astype(np.uint8))*255 for i in idx]
diam=18.0  # ~ median nucleus diameter in PanNuke at 40x (0.25 um/px)
labs={f:model.eval([downscale_local_mean(g,(f,f)) if f>1 else g for g in imgs],diameter=diam/f,channels=[0,0],batch_size=32)[0] for f in [1,2,4]}
rows=[]
for n in range(len(imgs)):
    L1=labs[1][n]
    for p in regionprops(L1):
        if p.area<30 or p.bbox[0]==0 or p.bbox[1]==0 or p.bbox[2]>=256 or p.bbox[3]>=256: continue
        rec={'img':int(idx[n]),'id':p.label}; ok=True
        for f in [1,2,4]:
            L=labs[f][n]
            if f==1: m=(L1==p.label)
            else:
                c=downscale_local_mean((L1==p.label).astype(float),(f,f))>=0.5
                ids,cnt=np.unique(L[c],return_counts=True); cnt=cnt[ids>0]; ids=ids[ids>0]
                if len(ids)==0: ok=False; break
                m=(L==ids[np.argmax(cnt)]); 
                if (m&c).sum()/(m|c).sum()<0.5: ok=False; break
            ys,xs=np.nonzero(m)
            if ys.min()==0 or xs.min()==0 or ys.max()==m.shape[0]-1 or xs.max()==m.shape[1]-1: ok=False; break
            cm=np.pad(m[ys.min():ys.max()+1,xs.min():xs.max()+1],2).astype(np.uint8)
            A=cm.sum()*f*f
            for k,fn in [('sk',p_skimage),('cv',p_opencv),('ij',p_imagej),('cr',p_crofton),('ours',p_corrected),('rc',lambda c: crofton_rc(c))]:
                rec[f'FF_{k}_{f}']=4*np.pi*A/(fn(cm)*f)**2
            rec[f'A_{f}']=A
        if ok: rows.append(rec)
df=pd.DataFrame(rows); df.to_csv(out,index=False)
print('images',len(imgs),'matched nuclei',len(df))
for k in ['sk','cv','ij','cr','ours','rc']:
    a1,a2,a4=df[f'FF_{k}_1'],df[f'FF_{k}_2'],df[f'FF_{k}_4']; sd=a1.std()
    print(f"{k:5s} mean {a1.mean():.3f} {a2.mean():.3f} {a4.mean():.3f} | shift/SD {((a2-a1).mean()/sd):+.2f} {((a4-a1).mean()/sd):+.2f}")
