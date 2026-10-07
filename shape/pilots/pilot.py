import numpy as np, cv2, time
from skimage.measure import regionprops, label, perimeter, perimeter_crofton, find_contours
rng=np.random.default_rng(0)
def ellipse(a,b,th,off,pad=4):
    R=int(np.ceil(max(a,b)))+pad; y,x=np.mgrid[-R:R+1,-R:R+1].astype(float)
    x-=off[0]; y-=off[1]; c,s=np.cos(th),np.sin(th)
    u=c*x+s*y; v=-s*x+c*y
    return ((u/a)**2+(v/b)**2<=1).astype(np.uint8)
def ests(m):
    A=m.sum()
    P={}
    P['sk4']=perimeter(m,4); P['sk8']=perimeter(m,8)
    P['crof4']=perimeter_crofton(m,4)
    cs,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); P['cv']=cv2.arcLength(cs[0],True)
    c=find_contours(np.pad(m,1).astype(float),0.5)[0]; P['msq']=np.sum(np.hypot(*np.diff(c,axis=0).T))
    return A,P
for r in [2,3,5,8,12,20,35,60]:
    res={}
    for t in range(40):
        m=ellipse(r,r,0,rng.uniform(-.5,.5,2))
        A,P=ests(m)
        for k,v in P.items(): res.setdefault(k,[]).append((v/(2*np.pi*r), 4*np.pi*A/v**2))
    print(r, '  '.join(f"{k}: P/Ptrue={np.mean([a for a,b in v]):.3f} circ={np.mean([b for a,b in v]):.3f}" for k,v in res.items()))
