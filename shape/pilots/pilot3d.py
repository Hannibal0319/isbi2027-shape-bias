import numpy as np
from skimage.measure import marching_cubes, mesh_surface_area
rng=np.random.default_rng(2)
def ellipsoid(a,b,c,R,off):
    n=int(np.ceil(max(a,b,c)))+3; z,y,x=np.mgrid[-n:n+1,-n:n+1,-n:n+1].astype(float)
    P=np.stack([x-off[0],y-off[1],z-off[2]],-1)@R.T
    return ((P[...,0]/a)**2+(P[...,1]/b)**2+(P[...,2]/c)**2<=1)
def randrot():
    q=rng.normal(size=4); q/=np.linalg.norm(q); w,x,y,z=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],[2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],[2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])
def crofton3d(m):
    # Cauchy-Crofton: S = 4 * mean over directions of (projected area) ; projected area ~ intersections count * area per line
    m=np.pad(m,2).astype(np.int8); dirs=[];W=[]
    import itertools
    for d in itertools.product([-1,0,1],repeat=3):
        if d<=(0,0,0): continue
        dirs.append(d)
    tot=0; S=[]
    for d in dirs:
        sh=np.roll(m,shift=d,axis=(2,1,0))
        n=np.sum((m==1)&(sh==0))  # exits along direction d
        L=np.linalg.norm(d)
        S.append(n/L)  # line density: lines along d have density 1/|d| per unit area... n*|d|^{-1}? approx
    return np.array(S),dirs
for r in [3,5,8,12,20,32]:
    res=[]
    for t in range(15):
        m=ellipsoid(r,r,r,randrot(),rng.uniform(-.5,.5,3))
        v,f,_,_=marching_cubes(np.pad(m,1).astype(float),0.5)
        A=mesh_surface_area(v,f); V=m.sum()
        Atrue=4*np.pi*r**2
        sph=(36*np.pi*V**2)**(1/3)/A
        faces=sum(np.sum(np.diff(np.pad(m,1).astype(np.int8),axis=k)!=0) for k in range(3))
        res.append([A/Atrue-1, sph, V/(4/3*np.pi*r**3)-1, faces/Atrue-1])
    r_=np.array(res).mean(0); print(r, "MC area err %+.3f  sphericity(MC) %.3f  vol err %+.3f  voxelface area err %+.3f"%tuple(r_))
