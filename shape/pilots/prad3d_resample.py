import numpy as np, SimpleITK as sitk, logging, sys
sys.path.insert(0,'.')
from radiomics import shape
from crofton3d import sphericity as sph_crofton
from scipy.ndimage import zoom
logging.getLogger('radiomics').setLevel(logging.ERROR)
rng=np.random.default_rng(0)
def sphere(R,sz,off):
    nz=int(np.ceil(R/sz))+3; n=int(np.ceil(R))+3
    z,y,x=np.meshgrid(np.arange(-nz,nz+1)*sz,np.arange(-n,n+1),np.arange(-n,n+1),indexing='ij')
    return ((x-off[0])**2+(y-off[1])**2+(z-off[2]*sz)**2<=R*R).astype(np.uint8)
def prad(m,sp):
    img=sitk.GetImageFromArray(np.ones(m.shape,np.float32)); img.SetSpacing(sp[::-1]); msk=sitk.GetImageFromArray(m.astype(np.uint8)); msk.SetSpacing(sp[::-1])
    return shape.RadiomicsShape(img,msk).getSphericityFeatureValue()
print("R slice | PyRad native | PyRad iso-resampled(linear,0.5) | Crofton native")
for R in [5,10,20]:
    for sz in [1,2,3,5]:
        a=[];b=[];c=[]
        for t in range(6):
            m=sphere(R,sz,rng.uniform(-.5,.5,3))
            a.append(prad(m,(sz,1,1)))
            mi=(zoom(m.astype(float),(sz,1,1),order=1)>=0.5).astype(np.uint8)
            b.append(prad(mi,(1,1,1)))
            c.append(sph_crofton(m,(sz,1,1)))
        print(R,sz,"| %.3f | %.3f | %.3f"%(np.mean(a),np.mean(b),np.mean(c)))
