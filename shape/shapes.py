import numpy as np
from matplotlib.path import Path
def polygon(kind, req, rng, n=2048):
    """Dense closed polygon (n,2) of a random shape with equivalent radius req (area = pi req^2)."""
    t=np.linspace(0,2*np.pi,n,endpoint=False)
    if kind.startswith('ell'):
        q=float(kind[3:]); a=1/np.sqrt(q); b=np.sqrt(q)
        x,y=a*np.cos(t),b*np.sin(t)
    else:  # random smooth star-shaped 'nucleus'
        rr=np.ones_like(t)
        for k in range(2,7):
            amp=rng.normal(0,0.10/k**0.8) if kind=='fourier' else rng.normal(0,0.25/k**0.7)
            rr+=amp*np.cos(k*t+rng.uniform(0,2*np.pi))
        rr=np.clip(rr,0.3,None); x,y=rr*np.cos(t),rr*np.sin(t)
    P=np.stack([x,y],1)
    A=0.5*abs(np.sum(P[:,0]*np.roll(P[:,1],-1)-np.roll(P[:,0],-1)*P[:,1]))
    P*=req/np.sqrt(A/np.pi)
    th=rng.uniform(0,2*np.pi); R=np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]])
    return P@R.T+rng.uniform(-.5,.5,2)
def truth(P):
    A=0.5*abs(np.sum(P[:,0]*np.roll(P[:,1],-1)-np.roll(P[:,0],-1)*P[:,1]))
    L=np.sum(np.linalg.norm(np.roll(P,-1,0)-P,axis=1)); return A,L
def rasterize(P,pad=3):
    lo=np.floor(P.min(0))-pad; hi=np.ceil(P.max(0))+pad
    xs=np.arange(lo[0],hi[0]+1); ys=np.arange(lo[1],hi[1]+1)
    X,Y=np.meshgrid(xs,ys); inside=Path(P).contains_points(np.stack([X.ravel(),Y.ravel()],1))
    return inside.reshape(X.shape).astype(np.uint8)
