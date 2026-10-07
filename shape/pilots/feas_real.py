import numpy as np, glob, os, cv2, sys
from skimage.io import imread
from skimage.measure import regionprops, perimeter, perimeter_crofton
from scipy.stats import spearmanr
root=sys.argv[1]; rng=np.random.default_rng(0)
dirs=sorted(glob.glob(root+'/*'))
rng.shuffle(dirs); dirs=dirs[:200]
def down(m,f):
    H,W=m.shape; H2,W2=H//f*f,W//f*f; m=m[:H2,:W2].astype(float)
    return (m.reshape(H2//f,f,W2//f,f).mean((1,3))>=0.5).astype(np.uint8)
def feats(m):
    if m.sum()<4: return None
    p=regionprops(m)[0]; A=p.area
    if p.euler_number!=1: return None
    return dict(A=A, ff_sk=4*np.pi*A/p.perimeter**2, ff_cr=4*np.pi*A/p.perimeter_crofton**2, sol=p.solidity)
rows=[]
for d in dirs:
    for f in glob.glob(d+'/masks/*.png'):
        m=(imread(f)>0).astype(np.uint8)
        ys,xs=np.nonzero(m); 
        if len(ys)<30: continue
        y0,x0=ys.min()//4*4,xs.min()//4*4
        m=np.pad(m[y0:ys.max()+9, x0:xs.max()+9],0)
        F=[feats(m),feats(down(m,2)),feats(down(m,4))]
        if any(x is None for x in F): continue
        rows.append(F)
print("nuclei:",len(rows))
for k in ['ff_sk','ff_cr','sol']:
    a=np.array([[r[i][k] for i in range(3)] for r in rows])
    d2=a[:,1]-a[:,0]; d4=a[:,2]-a[:,0]
    sd=a[:,0].std()
    print(f"{k}: mean@1x={a[:,0].mean():.3f} @1/2={a[:,1].mean():.3f} @1/4={a[:,2].mean():.3f} | paired shift/SD: 1/2: {d2.mean()/sd:+.2f}  1/4: {d4.mean()/sd:+.2f} | rho(area,feat)@1x={spearmanr([r[0]['A'] for r in rows],a[:,0])[0]:+.2f}")
