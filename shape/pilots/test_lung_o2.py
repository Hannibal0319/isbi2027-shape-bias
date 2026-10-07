import numpy as np, SimpleITK as sitk, glob, sys
from scipy.ndimage import label as cc, zoom, gaussian_filter, map_coordinates
from skimage.measure import marching_cubes, mesh_surface_area
from runcorr import crofton
rng=np.random.default_rng(1); FINE=0.5; SIG=1.5
out={1:{},2:{}}
files=sorted(glob.glob(sys.argv[1]+'/labelsTr/lung_*.nii.gz'))[::3]
for f in files:
    im=sitk.ReadImage(f); sp=np.array(im.GetSpacing()[::-1]); L=sitk.GetArrayFromImage(im)>0
    lab,n=cc(L,structure=np.ones((3,3,3)))
    for j in range(1,n+1):
        c=lab==j
        if c.sum()*np.prod(sp)<300: continue
        zz,yy,xx=np.nonzero(c); crop=np.pad(c[zz.min():zz.max()+1,yy.min():yy.max()+1,xx.min():xx.max()+1],4).astype(np.float32)
        fine=gaussian_filter(zoom(crop,sp/FINE,order=1),SIG/FINE)
        if fine.max()<0.6: continue
        v,fa,_,_=marching_cubes(fine,0.5,spacing=(FINE,)*3); A=mesh_surface_area(v,fa)
        Vt=abs(np.sum(np.einsum('ij,ij->i',v[fa[:,0]],np.cross(v[fa[:,1]],v[fa[:,2]]))))/6; true=(36*np.pi*Vt**2)**(1/3)/A
        ext=np.array(fine.shape)*FINE
        for s in [2.5,5.0]:
            tsp=np.array([s,0.7,0.7]); off=rng.uniform(0,1,3)*tsp
            Z,Y,X=np.meshgrid(*[np.arange(o,e,t) for o,e,t in zip(off,ext,tsp)],indexing='ij')
            m=np.pad(map_coordinates(fine,[Z/FINE,Y/FINE,X/FINE],order=1)>=0.5,1); V=m.sum()*np.prod(tsp)
            for o in [1,2]: out[o].setdefault(s,[]).append((36*np.pi*V**2)**(1/3)/crofton(m,tuple(tsp),True,o)-true)
for o in [1,2]: print('order',o,{s:(round(np.mean(v),3),round(np.std(v),3),len(v)) for s,v in out[o].items()})
