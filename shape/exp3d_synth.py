import numpy as np, pandas as pd, SimpleITK as sitk, logging
from radiomics import shape
from crofton3d import sphericity as sph_crofton
from scipy.ndimage import zoom
logging.getLogger('radiomics').setLevel(logging.ERROR)
rng=np.random.default_rng(7)
def randrot():
    q=rng.normal(size=4); q/=np.linalg.norm(q); w,x,y,z=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],[2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],[2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])
def ellipsoid(axes,sz,R,off):
    a=max(axes); nz=int(np.ceil(a/sz))+3; n=int(np.ceil(a))+3
    z,y,x=np.meshgrid(np.arange(-nz,nz+1)*sz-off[2]*sz,np.arange(-n,n+1)-off[1],np.arange(-n,n+1)-off[0],indexing='ij')
    P=np.stack([x,y,z],-1)@R
    return ((P[...,0]/axes[0])**2+(P[...,1]/axes[1])**2+(P[...,2]/axes[2])**2<=1).astype(np.uint8)
def prad(m,sp):
    img=sitk.GetImageFromArray(np.ones(m.shape,np.float32)); img.SetSpacing(sp[::-1]); msk=sitk.GetImageFromArray(m); msk.SetSpacing(sp[::-1])
    return shape.RadiomicsShape(img,msk).getSphericityFeatureValue()
from scipy.special import ellipkinc, ellipeinc
def ell_area(a,b,c):
    a,b,c=sorted([a,b,c],reverse=True)
    if a-c<1e-9: return 4*np.pi*a*a
    phi=np.arccos(c/a); k2=(a*a*(b*b-c*c))/(b*b*(a*a-c*c))
    return 2*np.pi*c*c+2*np.pi*a*b/np.sin(phi)*(ellipeinc(phi,k2)*np.sin(phi)**2+ellipkinc(phi,k2)*np.cos(phi)**2)
rows=[]
for shape_kind,ratio in [('sphere',(1,1,1)),('ellipsoid',(1,0.75,0.5))]:
  for Req in [3,5,10,20]:
    for sz in [1,1.5,2,2.5,3,4,5]:
      for t in range(10):
        s=Req/np.prod(ratio)**(1/3); axes=tuple(s*r for r in ratio)
        R=randrot(); m=ellipsoid(axes,sz,R,rng.uniform(-.5,.5,3))
        if m.sum()<5: continue
        true=(36*np.pi*(4/3*np.pi*np.prod(axes))**2)**(1/3)/ell_area(*axes)
        mi=(zoom(m.astype(float),(sz,1,1),order=1)>=0.5).astype(np.uint8)
        rows.append(dict(kind=shape_kind,R=Req,slice=sz,true=true,prad=prad(m,(sz,1,1)),prad_iso=prad(mi,(1,1,1)),crofton=sph_crofton(m,(sz,1,1))))
pd.DataFrame(rows).to_csv('exp3d_synth.csv',index=False)
d=pd.DataFrame(rows); d['e_prad']=d.prad-d.true; d['e_iso']=d.prad_iso-d.true; d['e_cr']=d.crofton-d.true
print(d.groupby(['kind','R','slice'])[['true','e_prad','e_iso','e_cr']].mean().round(3).to_string())
