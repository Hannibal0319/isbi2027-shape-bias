import numpy as np, SimpleITK as sitk, logging, sys
from radiomics import shape
logging.getLogger('radiomics').setLevel(logging.ERROR)
rng=np.random.default_rng(0)
def sphere(R,sz,off,ax=None):
    # grid spacing (1,1,sz) mm in (x,y,z); ellipsoid axes ax (mm) random rotation
    nz=int(np.ceil(R/sz))+3; n=int(np.ceil(R))+3
    z,y,x=np.meshgrid(np.arange(-nz,nz+1)*sz,np.arange(-n,n+1),np.arange(-n,n+1),indexing='ij')
    return ((x-off[0])**2+(y-off[1])**2+(z-off[2]*sz)**2<=R*R).astype(np.uint8)
print("R(mm)  slice(mm): sphericity mean±sd (PyRadiomics)")
for R in [4,6,10,15,20]:
    line=[]
    for sz in [1,2,3,5]:
        v=[]
        for t in range(8):
            m=sphere(R,sz,rng.uniform(-.5,.5,3))
            img=sitk.GetImageFromArray(np.ones(m.shape,np.float32)); img.SetSpacing((1,1,sz))
            msk=sitk.GetImageFromArray(m); msk.SetSpacing((1,1,sz))
            f=shape.RadiomicsShape(img,msk); v.append(f.getSphericityFeatureValue())
        line.append(f"{np.mean(v):.3f}±{np.std(v):.3f}")
    print(R, "  ".join(line))
