import numpy as np, cv2
from skimage.measure import regionprops, perimeter, perimeter_crofton
from scipy.special import ellipe
rng=np.random.default_rng(1)
def ellipse(a,b,th,off,pad=4):
    R=int(np.ceil(max(a,b)))+pad; y,x=np.mgrid[-R:R+1,-R:R+1].astype(float)
    x-=off[0]; y-=off[1]; c,s=np.cos(th),np.sin(th); u=c*x+s*y; v=-s*x+c*y
    return ((u/a)**2+(v/b)**2<=1).astype(np.uint8)
def trueP(a,b): return 4*a*ellipe(1-(b/a)**2)
print("a  b/a | rel.err mean±sd over rotations: P_sk4 P_crof P_cv | minor_sk minor_shep | ecc_sk ecc_shep | solidity_sk")
for ratio in [1,0.5,0.25]:
  for a in [3,6,12,24,48]:
    b=a*ratio; rows=[]
    for t in range(60):
        th=rng.uniform(0,np.pi); m=ellipse(a,b,th,rng.uniform(-.5,.5,2))
        p=regionprops(m)[0]; P=trueP(a,b)
        cs,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        l1,l2=p.inertia_tensor_eigvals
        e_true=np.sqrt(1-ratio**2)
        rows.append([p.perimeter/P-1, p.perimeter_crofton/P-1, cv2.arcLength(cs[0],True)/P-1,
                     p.axis_minor_length/(2*b)-1, 4*np.sqrt(l2+1/12)/(2*b)-1,
                     p.eccentricity-e_true, np.sqrt(1-(l2+1/12)/(l1+1/12))-e_true, p.solidity-1])
    r=np.array(rows); print(f"{a:3d} {ratio:.2f} | "+"  ".join(f"{m:+.3f}±{s:.3f}" for m,s in zip(r.mean(0),r.std(0))))
