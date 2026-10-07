import sys, numpy as np, SimpleITK as sitk, logging
from radiomics import shape2D
logging.getLogger('radiomics').setLevel(logging.ERROR)
d=np.load(sys.argv[1]); out=[]
for i in range(len(d.files)):
    m=np.pad(d[f'm{i}'],2)
    img=sitk.GetImageFromArray(np.ones((1,)+m.shape,np.float32)); msk=sitk.GetImageFromArray(m[None].astype(np.uint8))
    f=shape2D.RadiomicsShape2D(img,msk,force2D=True)
    out.append([f.getPerimeterFeatureValue(), f.getMeshSurfaceFeatureValue(), f.getSphericityFeatureValue()])
np.save(sys.argv[2],np.array(out))
