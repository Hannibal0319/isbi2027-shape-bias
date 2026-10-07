"""Cauchy-Crofton surface area of a 3D binary mask on an (an)isotropic voxel grid (13 lattice directions)."""
import numpy as np, itertools
from functools import lru_cache
DIRS=[d for d in itertools.product([-1,0,1],repeat=3) if d>(0,0,0)]  # 13 directions, zyx order

@lru_cache(maxsize=64)
def crofton_weights(spacing, n=2_000_000, seed=0):
    """Fraction of the unit sphere closest (up to sign) to each physical lattice direction (Voronoi solid angles)."""
    D=np.array(DIRS,float)*np.array(spacing); D/=np.linalg.norm(D,axis=1,keepdims=True)
    u=np.random.default_rng(seed).normal(size=(n,3)); u/=np.linalg.norm(u,axis=1,keepdims=True)
    idx=np.argmax(np.abs(u@D.T),axis=1)
    return np.bincount(idx,minlength=len(DIRS))/n

def surface_area(m, spacing=(1.,1.,1.)):
    """m: bool array (z,y,x); spacing: (sz,sy,sx). Returns surface area in physical units."""
    m=np.pad(np.asarray(m,bool),1); sp=np.array(spacing,float); V=np.prod(sp)
    w=crofton_weights(tuple(float(s) for s in spacing)); S=0.0
    for wd,d in zip(w,DIRS):
        sl_a=tuple(slice(0,n-max(k,0)+min(k,0) if True else None) for n,k in zip(m.shape,d))
        # exits: voxel p in object, p+d not in object
        a=m[tuple(slice(max(-k,0), m.shape[i]-max(k,0)) for i,k in enumerate(d))]
        b=m[tuple(slice(max(k,0), m.shape[i]-max(-k,0)) for i,k in enumerate(d))]
        exits=np.count_nonzero(a & ~b)
        S+=wd*exits*V/np.linalg.norm(np.array(d)*sp)
    return 4*S

def sphericity(m, spacing=(1.,1.,1.)):
    V=np.count_nonzero(m)*np.prod(spacing); S=surface_area(m,spacing)
    return (36*np.pi*V**2)**(1/3)/S
