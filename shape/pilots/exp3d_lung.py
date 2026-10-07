"""Real 3D: sphericity of MSD lung tumours at native vs re-digitised thicker slices (PyRadiomics vs Crofton)."""
import numpy as np, pandas as pd, SimpleITK as sitk, logging, glob, sys, os
from radiomics import shape
from scipy.ndimage import label as cc
from crofton3d import sphericity as sph_crofton
logging.getLogger('radiomics').setLevel(logging.ERROR)
def prad(m,sp):
    img=sitk.GetImageFromArray(np.ones(m.shape,np.float32)); img.SetSpacing(tuple(float(s) for s in sp[::-1]))
    msk=sitk.GetImageFromArray(m.astype(np.uint8)); msk.SetSpacing(tuple(float(s) for s in sp[::-1]))
    return shape.RadiomicsShape(img,msk).getSphericityFeatureValue()
def thicken(m,k,phase):
    m=m[phase:]; Z=m.shape[0]//k*k; m=m[:Z].astype(float)
    return (m.reshape(Z//k,k,*m.shape[1:]).mean(1)>=0.5).astype(np.uint8)
rows=[]
for f in sorted(glob.glob(sys.argv[1]+'/labelsTr/lung_*.nii.gz')):
    im=sitk.ReadImage(f); sp=im.GetSpacing()[::-1]; L=sitk.GetArrayFromImage(im)>0
    lab,n=cc(L,structure=np.ones((3,3,3)))
    for j in range(1,n+1):
        c=lab==j
        if c.sum()*np.prod(sp)<100: continue  # >= 100 mm^3
        zz,yy,xx=np.nonzero(c); crop=c[max(zz.min()-2,0):zz.max()+3, max(yy.min()-2,0):yy.max()+3, max(xx.min()-2,0):xx.max()+3]
        rec=dict(case=os.path.basename(f),comp=j,sz=sp[0],sxy=sp[1],vol=c.sum()*np.prod(sp),
                 Req=(3*c.sum()*np.prod(sp)/(4*np.pi))**(1/3))
        rec['prad_native']=prad(crop,sp); rec['cr_native']=sph_crofton(crop,sp)
        for T in [2.5,5.0]:
            k=int(round(T/sp[0]))
            if k<2: rec[f'prad_{T}']=rec[f'cr_{T}']=np.nan; continue
            pv=[];cv=[]
            for ph in range(k):
                t=thicken(crop,k,ph)
                if t.sum()<3: continue
                spk=(sp[0]*k,sp[1],sp[2]); pv.append(prad(np.pad(t,((1,1),(0,0),(0,0))),spk)); cv.append(sph_crofton(t,spk))
            rec[f'prad_{T}']=np.mean(pv) if pv else np.nan; rec[f'cr_{T}']=np.mean(cv) if cv else np.nan
        rows.append(rec); print(rec['case'],round(rec['sz'],2),round(rec['Req'],1),{k:round(v,3) for k,v in rec.items() if k.startswith(('prad','cr'))},flush=True)
pd.DataFrame(rows).to_csv('exp3d_lung.csv',index=False)
