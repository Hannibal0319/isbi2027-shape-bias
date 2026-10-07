"""Realistic 3D phantoms from MSD lung tumours: smooth continuous surface (known truth), digitized at several slice thicknesses."""
import numpy as np, pandas as pd, SimpleITK as sitk, logging, glob, sys, os
from radiomics import shape
from scipy.ndimage import label as cc, zoom, gaussian_filter, map_coordinates
from skimage.measure import marching_cubes, mesh_surface_area
from crofton3d import sphericity as sph_crofton
logging.getLogger('radiomics').setLevel(logging.ERROR)
rng=np.random.default_rng(0); FINE=0.5; SIG=1.5; INPLANE=0.7; SLICES=[0.7,1.25,2.5,3.75,5.0]
def prad(m,sp):
    img=sitk.GetImageFromArray(np.ones(m.shape,np.float32)); img.SetSpacing(tuple(float(s) for s in sp[::-1]))
    msk=sitk.GetImageFromArray(m.astype(np.uint8)); msk.SetSpacing(tuple(float(s) for s in sp[::-1]))
    return shape.RadiomicsShape(img,msk).getSphericityFeatureValue()
rows=[]
for f in sorted(glob.glob(sys.argv[1]+'/labelsTr/lung_*.nii.gz')):
    im=sitk.ReadImage(f); sp=np.array(im.GetSpacing()[::-1]); L=sitk.GetArrayFromImage(im)>0
    lab,n=cc(L,structure=np.ones((3,3,3)))
    for j in range(1,n+1):
        c=lab==j
        if c.sum()*np.prod(sp)<300: continue
        zz,yy,xx=np.nonzero(c); pad=4
        crop=np.pad(c[zz.min():zz.max()+1,yy.min():yy.max()+1,xx.min():xx.max()+1],pad).astype(np.float32)
        # continuous field on fine isotropic grid
        fine=gaussian_filter(zoom(crop,sp/FINE,order=1),SIG/FINE)
        if fine.max()<0.6: continue
        v,fa,_,_=marching_cubes(fine,0.5,spacing=(FINE,)*3); A=mesh_surface_area(v,fa)
        Vtrue=abs(np.sum(np.einsum('ij,ij->i',v[fa[:,0]],np.cross(v[fa[:,1]],v[fa[:,2]]))))/6
        true=(36*np.pi*Vtrue**2)**(1/3)/A
        ext=np.array(fine.shape)*FINE
        rec=dict(case=os.path.basename(f),comp=j,Req=(3*Vtrue/(4*np.pi))**(1/3),true=true)
        for s in SLICES:
            tsp=np.array([s,INPLANE,INPLANE]); pv=[];cv=[]
            for rep in range(3):
                off=rng.uniform(0,1,3)*tsp
                axes=[np.arange(o,e,t) for o,e,t in zip(off,ext,tsp)]
                Z,Y,X=np.meshgrid(*axes,indexing='ij')
                m=map_coordinates(fine,[Z/FINE,Y/FINE,X/FINE],order=1)>=0.5
                if m.sum()<5: continue
                m=np.pad(m,1); pv.append(prad(m,tsp)); cv.append(sph_crofton(m,tsp))
            rec[f'prad_{s}']=np.mean(pv); rec[f'cr_{s}']=np.mean(cv)
        rows.append(rec); print(rec['case'],round(rec['Req'],1),round(true,3),' '.join(f"{s}:{rec[f'prad_{s}']-true:+.3f}/{rec[f'cr_{s}']-true:+.3f}" for s in SLICES),flush=True)
pd.DataFrame(rows).to_csv('exp3d_lung_phantom.csv',index=False)
d=pd.DataFrame(rows)
for s in SLICES: print(s,'PyRad bias %.3f±%.3f  Crofton bias %.3f±%.3f'%((d[f'prad_{s}']-d.true).mean(),(d[f'prad_{s}']-d.true).std(),(d[f'cr_{s}']-d.true).mean(),(d[f'cr_{s}']-d.true).std()))
