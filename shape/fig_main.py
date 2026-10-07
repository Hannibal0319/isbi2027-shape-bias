import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import gaussian_kde
import cv2
from skimage.measure import find_contours
from estimators import KAPPA, _pixel_edge_polygon
plt.rcParams.update({'font.size':6.5,'font.family':'serif','axes.linewidth':0.6,'xtick.major.width':0.6,'ytick.major.width':0.6,
                     'xtick.major.size':2.5,'ytick.major.size':2.5,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'legend.fontsize':5.6})
C=dict(blue='#2a78d6',orange='#eb6834',aqua='#1baf7a',magenta='#e87ba4',violet='#4a3aa7',red='#e34948',grey='#7a7974')
fig,ax=plt.subplots(1,5,figsize=(7.1,1.45),gridspec_kw=dict(wspace=0.42,width_ratios=[0.8,1.4,0.66,0.66,0.98]))
# (a) schematic: digitized disk, true boundary and traced polygons
s=ax[0]; R=3.3; c0=np.array([0.27,0.18]); n=6
yy,xx=np.mgrid[-n:n+1,-n:n+1]; m=(((xx-c0[0])**2+(yy-c0[1])**2)<=R*R).astype(np.uint8)
for y,x in zip(*np.nonzero(m)): s.add_patch(Rectangle((x-n-0.5,y-n-0.5),1,1,fc='#e4e3df',ec='white',lw=0.4))
t=np.linspace(0,2*np.pi,200); s.plot(c0[0]+R*np.cos(t),c0[1]+R*np.sin(t),'k--',lw=0.7,label='true boundary')
cs,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); q=cs[0][:,0,:]-n; q=np.vstack([q,q[:1]])
s.plot(q[:,0],q[:,1],color=C['blue'],lw=1.0,label='chain code')
ms=find_contours(np.pad(m,1).astype(float),0.5)[0]; s.plot(ms[:,1]-1-n,ms[:,0]-1-n,color=C['magenta'],lw=1.0,label='marching sq.')
pe=_pixel_edge_polygon(m)-1-n-0.5; pe=np.vstack([pe,pe[:1]]); s.plot(pe[:,0],pe[:,1],color=C['violet'],lw=0.9,label='pixel edges')
s.set_aspect('equal'); s.set_xlim(-4.8,4.8); s.set_ylim(-8.2,4.6); s.axis('off')
s.legend(loc='lower center',bbox_to_anchor=(0.38,0.0),frameon=False,handlelength=1.3,borderaxespad=0,labelspacing=0.12,ncol=1,columnspacing=0.6,fontsize=5.4)
s.set_title('(a) disk, $r$=3.3 px',fontsize=6.5)
# (b) synthetic 2D bias
df=pd.read_csv('exp1_synth_rc.csv'); a=ax[1]
series=[('P_skimage/CellProfiler','skimage/CellProf.',C['blue'],'o'),
        ('P_ImageJ','ImageJ',C['aqua'],'^'),('P_PyRadiomics','PyRadiomics',C['magenta'],'v'),('matlab','MATLAB R23a',C['violet'],'D'),
        ('P_Crofton (skimage)','Crofton',C['grey'],'x'),('P_corrected chain (ours)','F1 (ours)',C['orange'],'*'),('P_rc','RCC (ours)',C['red'],'P')]
for k,lab,c,mk in series:
    est=df.ff_matlab2023 if k=='matlab' else 4*np.pi*df.Ad/df[k]**2
    mm=(est-df.ff_true).groupby(df.r).mean(); a.plot(mm.index,mm.values,marker=mk,ms=2.2,lw=0.85,color=c,label=lab)
th_chain=(df.ff_true*((df.L/(KAPPA*(df.L-2*np.sqrt(2))))**2-1)).groupby(df.r).mean(); th_ms=(df.ff_true*(1/KAPPA**2-1)).groupby(df.r).mean()
r=th_chain.index[th_chain.index>=3]
a.plot(r,th_chain[r],'k--',lw=0.7,label='model (1)',zorder=5); a.plot(th_ms.index,th_ms.values,'k--',lw=0.7,zorder=5)
a.axhline(0,color='#bbb',lw=0.5,zorder=0); a.set_xscale('log'); a.set_xticks([2,4,8,16,32]); a.set_xticklabels([2,4,8,16,32])
a.set_xlabel('equivalent radius (px)'); a.set_ylabel('circularity bias'); a.set_ylim(-0.12,0.78); a.set_yticks([0,0.2,0.4]); a.set_title('(b) 2D, synthetic',fontsize=6.5)
a.legend(loc='upper right',frameon=False,handlelength=1.2,borderaxespad=0,labelspacing=0.12,ncol=2,columnspacing=0.4,fontsize=4.9,handletextpad=0.3)
# (c,d) real distributions
Rr=pd.read_csv('reseg_bbbc_all.csv'); x=np.linspace(0.45,1.6,300); shades=['#9ec5f4','#2a78d6','#0b3e7e']
for a,k,tt in [(ax[2],'sk','(c) default (skimage)'),(ax[3],'ours','(d) F1 (ours)')]:
    for f,c,lab in zip([1,2,4],shades,['native','1/2','1/4']):
        v=Rr[f'FF_{k}_{f}'].values; v=v[(v>0.3)&(v<2)]
        a.plot(x,gaussian_kde(v)(x),color=c,lw=0.95,label=lab); a.axvline(np.median(v),color=c,lw=0.6,ls=':')
    a.axvline(1,color='#bbb',lw=0.5,zorder=0); a.set_yticks([]); a.spines['left'].set_visible(False)
    a.set_xlabel('form factor'); a.set_title(tt,fontsize=6.5); a.set_xlim(0.45,1.6)
ax[2].legend(title='resolution',title_fontsize=5.6,loc='upper right',frameon=False,handlelength=1.1,borderaxespad=0,labelspacing=0.15)
ymax=max(ax[2].get_ylim()[1],ax[3].get_ylim()[1]); ax[2].set_ylim(0,ymax); ax[3].set_ylim(0,ymax)
# (e) 3D
d=pd.read_csv('exp3d_synth.csv'); d=d[d.R>=5]; b=ax[4]
for col,lab,c,mk,ls in [('prad','PyRad. native',C['magenta'],'v','-'),('prad_iso','PyRad. iso-resampled',C['violet'],'D','--'),('crofton','Crofton',C['grey'],'x','-'),('rcc','RCC (ours)',C['red'],'P','-')]:
    g=(d[col]-d.true).groupby(d.slice); b.plot(g.mean().index,g.mean().values,marker=mk,ms=2.2,lw=0.85,ls=ls,color=c,label=lab)
    b.fill_between(g.mean().index,g.quantile(0.1),g.quantile(0.9),color=c,alpha=0.15,lw=0)
b.axhline(0,color='#bbb',lw=0.5,zorder=0); b.set_xlabel('slice thickness (mm)'); b.set_ylabel('sphericity bias'); b.set_title(r'(e) 3D, synthetic, R$\geq$5 mm',fontsize=6.5)
b.set_ylim(-0.29,0.07); b.set_yticks([0,-0.1,-0.2]); b.legend(loc='lower center',ncol=2,frameon=False,handlelength=1.3,borderaxespad=0.1,labelspacing=0.15,columnspacing=0.6,fontsize=5.0,handletextpad=0.3)
fig.savefig('../paper/figures/fig_main.pdf',bbox_inches='tight'); fig.savefig('fig_main.png',dpi=250,bbox_inches='tight')
