"""Runtime of perimeter / surface estimators per object (median over repeats)."""
import time, numpy as np, SimpleITK as sitk, logging
from skimage.measure import perimeter, perimeter_crofton, regionprops
from runcorr import crofton as rcc
from radiomics import shape, shape2D
logging.getLogger('radiomics').setLevel(logging.ERROR)
def t(f, n=50):
    ts=[]
    for _ in range(n):
        a=time.perf_counter(); f(); ts.append(time.perf_counter()-a)
    return np.median(ts)*1e3
print('2D nucleus-like disks (ms per object)')
for r in [5, 15, 40]:
    y,x=np.mgrid[-r-3:r+4,-r-3:r+4]; m=(x*x+y*y<=r*r)
    img=sitk.GetImageFromArray(np.ones((1,)+m.shape,np.float32)); msk=sitk.GetImageFromArray(m[None].astype(np.uint8))
    print(f' r={r:3d}: skimage perimeter {t(lambda: perimeter(m,4)):.3f} | crofton {t(lambda: perimeter_crofton(m,4)):.3f} | RCC {t(lambda: rcc(m)):.3f} | PyRadiomics shape2D {t(lambda: shape2D.RadiomicsShape2D(img,msk,force2D=True).getPerimeterFeatureValue(),10):.3f}')
print('3D lesions, spacing (5,0.7,0.7) mm (ms per object)')
for R in [5, 15, 40]:
    sp=(5.0,0.7,0.7); nz=int(R/5)+3; n=int(R/0.7)+3
    z,y,x=np.meshgrid(np.arange(-nz,nz+1)*5.0,np.arange(-n,n+1)*0.7,np.arange(-n,n+1)*0.7,indexing='ij'); m=(x*x+y*y+z*z<=R*R)
    img=sitk.GetImageFromArray(np.ones(m.shape,np.float32)); img.SetSpacing(sp[::-1]); msk=sitk.GetImageFromArray(m.astype(np.uint8)); msk.SetSpacing(sp[::-1])
    print(f' R={R:2d} mm ({m.sum()} vox): RCC {t(lambda: rcc(m,sp),10):.1f} | PyRadiomics shape (mesh) {t(lambda: shape.RadiomicsShape(img,msk).getSphericityFeatureValue(),5):.1f}')
