"""Independent re-segmentation at native and downsampled resolution with Cellpose; matched-nucleus FF shifts."""
import numpy as np, glob, sys, pandas as pd
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import downscale_local_mean
from skimage.measure import regionprops
from cellpose import models
sys.path.insert(0,'.')
from estimators import p_skimage, p_crofton, p_corrected, p_opencv, p_imagej
N=int(sys.argv[1]); out=sys.argv[2]
model=models.Cellpose(gpu=True,model_type='nuclei')
dirs=sorted(glob.glob('../data/bbbc038/train/*'))
rows=[]; used=0
for d in dirs:
    im=imread(glob.glob(d+'/images/*.png')[0])[...,:3]
    if not (np.allclose(im[...,0],im[...,1]) and np.allclose(im[...,1],im[...,2])): continue  # fluorescence (grey) only
    g=im[...,0].astype(float); H,W=g.shape; g=g[:H//4*4,:W//4*4]
    gt=[ (imread(f)>0) for f in glob.glob(d+'/masks/*.png')]
    diam=np.median([2*np.sqrt(m.sum()/np.pi) for m in gt])
    if diam<12: continue
    labs={}
    for f in [1,2,4]:
        x=downscale_local_mean(g,(f,f)) if f>1 else g
        lab,_,_,_=model.eval(x,diameter=diam/f,channels=[0,0])
        labs[f]=lab
    used+=1
    # match: for each native object, find object at scale f with max IoU (compare on coarse grid)
    for p in regionprops(labs[1]):
        if p.area<30 or p.bbox[0]==0 or p.bbox[1]==0 or p.bbox[2]>=labs[1].shape[0] or p.bbox[3]>=labs[1].shape[1]: continue
        rec={'img':d[-12:],'id':p.label}
        ok=True
        for f in [1,2,4]:
            L=labs[f]
            if f==1: m=(L==p.label)
            else:
                nat=(labs[1]==p.label); c=downscale_local_mean(nat.astype(float),(f,f))>=0.5
                ids,cnt=np.unique(L[c],return_counts=True); cnt=cnt[ids>0]; ids=ids[ids>0]
                if len(ids)==0: ok=False; break
                j=ids[np.argmax(cnt)]; m=(L==j); iou=(m&c).sum()/(m|c).sum()
                if iou<0.5: ok=False; break
            ys,xs=np.nonzero(m)
            if ys.min()==0 or xs.min()==0 or ys.max()==m.shape[0]-1 or xs.max()==m.shape[1]-1: ok=False; break
            cm=np.pad(m[ys.min():ys.max()+1,xs.min():xs.max()+1],2).astype(np.uint8)
            if cm.sum()<4: ok=False; break
            A=cm.sum()*f*f
            for k,fn in [('sk',p_skimage),('cv',p_opencv),('ij',p_imagej),('cr',p_crofton),('ours',p_corrected)]:
                rec[f'FF_{k}_{f}']=4*np.pi*A/(fn(cm)*f)**2
            rec[f'A_{f}']=A
        if ok: rows.append(rec)
    if used>=N: break
df=pd.DataFrame(rows); df.to_csv(out,index=False)
print('images',used,'matched nuclei',len(df))
for k in ['sk','cv','ij','cr','ours']:
    a1,a2,a4=df[f'FF_{k}_1'],df[f'FF_{k}_2'],df[f'FF_{k}_4']; sd=a1.std()
    print(f"{k:5s} mean {a1.mean():.3f} {a2.mean():.3f} {a4.mean():.3f} | shift/SD {((a2-a1).mean()/sd):+.2f} {((a4-a1).mean()/sd):+.2f}")
