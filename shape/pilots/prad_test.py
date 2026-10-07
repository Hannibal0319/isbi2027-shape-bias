import numpy as np, SimpleITK as sitk, logging
from radiomics import shape, shape2D, cShape
logging.getLogger('radiomics').setLevel(logging.ERROR)
rng=np.random.default_rng(0)
def run2d(m):
    img=sitk.GetImageFromArray(np.ones((1,)+m.shape,np.float32)); msk=sitk.GetImageFromArray(m[None].astype(np.uint8))
    f=shape2D.RadiomicsShape2D(img,msk,force2D=True); f.execute() if False else None
    return f.getPerimeterFeatureValue(), f.getPixelSurfaceFeatureValue(), f.getSphericityFeatureValue()
def run3d(m):
    img=sitk.GetImageFromArray(np.ones(m.shape,np.float32)); msk=sitk.GetImageFromArray(m.astype(np.uint8))
    f=shape.RadiomicsShape(img,msk)
    return f.getSurfaceAreaFeatureValue(), f.getMeshVolumeFeatureValue(), f.getSphericityFeatureValue()
for r in [2,3,5,8,12,20,40]:
    o=[];s=[]
    for t in range(20):
        off=rng.uniform(-.5,.5,3); n=r+4
        y,x=np.mgrid[-n:n+1,-n:n+1]; m=((x-off[0])**2+(y-off[1])**2<=r*r)
        P,A,S=run2d(m); o.append([P/(2*np.pi*r), S])
        if r<=20:
            z,y,x=np.mgrid[-n:n+1,-n:n+1,-n:n+1]; m3=((x-off[0])**2+(y-off[1])**2+(z-off[2])**2<=r*r)
            SA,V,Sp=run3d(m3); s.append([SA/(4*np.pi*r*r), V/(4/3*np.pi*r**3), Sp])
    print(r,"2D P/Ptrue %.3f  'Sphericity'(2D) %.3f"%tuple(np.mean(o,0)), ("| 3D SA/true %.3f MeshVol/true %.3f Sphericity %.3f"%tuple(np.mean(s,0))) if s else "")
