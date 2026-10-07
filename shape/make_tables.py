import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from scipy.stats import spearmanr
def sim(csv,prad):
    df=pd.read_csv(csv); pr=np.load(prad); df['P_PyRadiomics']=pr[:,0]*df.scale
    ests=[c[2:] for c in df.columns if c.startswith('P_')]
    for e in ests: df['FF_'+e]=4*np.pi*df.A/df['P_'+e]**2
    df['FF_MATLAB R2023a']=df['FF_skimage/CellProfiler']*(1-0.5/(df['P_skimage/CellProfiler']/df.scale/(2*np.pi)+0.5))**2
    key=['src','img','cls','iid']; W={s:df[df.scale==s].set_index(key) for s in [1,2,4]}
    common=W[1].index.intersection(W[2].index).intersection(W[4].index); W={s:W[s].loc[common] for s in W}
    out={}
    for e in ests+['MATLAB R2023a']:
        a=[W[s]['FF_'+e].values for s in [1,2,4]]; sd=a[0].std()
        # bootstrap SE of shift (nuclei resampled)
        out[e]=dict(s2=(a[1]-a[0]).mean()/sd, s4=(a[2]-a[0]).mean()/sd, rho=spearmanr(W[1].A,a[0])[0])
    return pd.DataFrame(out).T, len(common)
def reseg(csv):
    df=pd.read_csv(csv); out={}
    names={'sk':'skimage/CellProfiler','cv':'OpenCV','ij':'ImageJ','cr':'Crofton (skimage)','ours':'corrected chain (ours)'}
    for f in [1,2,4]:
        P=np.sqrt(4*np.pi*df[f'A_{f}']/df[f'FF_sk_{f}'])
        df[f'FF_mat_{f}']=df[f'FF_sk_{f}']*(1-0.5/(P/f/(2*np.pi)+0.5))**2
    names['mat']='MATLAB R2023a'
    for k,n in names.items():
        a=[df[f'FF_{k}_{f}'] for f in [1,2,4]]; sd=a[0].std()
        d2=(a[1]-a[0]); d4=(a[2]-a[0])
        out[n]=dict(r2=d2.mean()/sd, r4=d4.mean()/sd, se4=d4.std()/np.sqrt(len(df))/sd)
    return pd.DataFrame(out).T, len(df)
pn,npn=sim('real_pannuke_2656.csv','real_pannuke_2656_prad.npy')
bb,nbb=sim('real_bbbc_670.csv','real_bbbc_670_prad.npy')
rpn,nrpn=reseg('reseg_pn_1500.csv'); rbb,nrbb=reseg('reseg_bbbc_all.csv')
T=pd.concat({'PN sim':pn,'PN cellpose':rpn,'BB sim':bb,'BB cellpose':rbb},axis=1)
print(npn,nrpn,nbb,nrbb); print(T.round(2).to_string())
