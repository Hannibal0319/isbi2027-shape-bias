import numpy as np, pandas as pd
from runcorr import crofton
from scipy.special import ellipkinc, ellipeinc
rng=np.random.default_rng(7)
def randrot():
    q=rng.normal(size=4); q/=np.linalg.norm(q); w,x,y,z=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],[2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],[2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])
def ellipsoid(axes,sz,R,off):
    a=max(axes); nz=int(np.ceil(a/sz))+3; n=int(np.ceil(a))+3
    z,y,x=np.meshgrid(np.arange(-nz,nz+1)*sz-off[2]*sz,np.arange(-n,n+1)-off[1],np.arange(-n,n+1)-off[0],indexing='ij')
    P=np.stack([x,y,z],-1)@R
    return ((P[...,0]/axes[0])**2+(P[...,1]/axes[1])**2+(P[...,2]/axes[2])**2<=1)
def ell_area(a,b,c):
    a,b,c=sorted([a,b,c],reverse=True)
    if a-c<1e-9: return 4*np.pi*a*a
    phi=np.arccos(c/a); k2=(a*a*(b*b-c*c))/(b*b*(a*a-c*c))
    return 2*np.pi*c*c+2*np.pi*a*b/np.sin(phi)*(ellipeinc(phi,k2)*np.sin(phi)**2+ellipkinc(phi,k2)*np.cos(phi)**2)
rows=[]
for kind,ratio in [('sphere',(1,1,1)),('ellipsoid',(1,0.75,0.5))]:
  for Req in [3,5,10,20]:
    for sz in [1,2,3,5]:
      for t in range(8):
        s=Req/np.prod(ratio)**(1/3); axes=tuple(s*r for r in ratio); m=ellipsoid(axes,sz,randrot(),rng.uniform(-.5,.5,3))
        V=m.sum()*sz; true=(36*np.pi*(4/3*np.pi*np.prod(axes))**2)**(1/3)/ell_area(*axes)
        sp=(sz,1.,1.)
        f=lambda S:(36*np.pi*V**2)**(1/3)/S
        rows.append(dict(kind=kind,R=Req,slice=sz,e_cr=f(crofton(m,sp,False))-true,e_rc=f(crofton(m,sp,True))-true))
d=pd.DataFrame(rows); print(d.groupby(['R','slice'])[['e_cr','e_rc']].mean().round(3).unstack('slice').to_string())
