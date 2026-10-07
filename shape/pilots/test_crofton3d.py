import numpy as np
from crofton3d import sphericity, surface_area
rng=np.random.default_rng(0)
def sphere(R,sz,off):
    nz=int(np.ceil(R/sz))+3; n=int(np.ceil(R))+3
    z,y,x=np.meshgrid(np.arange(-nz,nz+1)*sz,np.arange(-n,n+1),np.arange(-n,n+1),indexing='ij')
    return ((x-off[0])**2+(y-off[1])**2+(z-off[2]*sz)**2<=R*R)
print("R  slice: sphericity Crofton3D (mean±sd); SA err")
for R in [2,3,4,6,10,15,20]:
    line=[]
    for sz in [1,2,3,5]:
        v=[];e=[]
        for t in range(8):
            m=sphere(R,sz,rng.uniform(-.5,.5,3)); v.append(sphericity(m,(sz,1,1))); e.append(surface_area(m,(sz,1,1))/(4*np.pi*R*R)-1)
        line.append(f"{np.mean(v):.3f}±{np.std(v):.3f} ({np.mean(e):+.3f})")
    print(R,"  ".join(line))
