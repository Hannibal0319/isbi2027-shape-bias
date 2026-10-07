import numpy as np, pandas as pd, sys, warnings; warnings.filterwarnings('ignore')
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from estimators import KAPPA
df=pd.read_csv(sys.argv[1])
if len(sys.argv)>2:
    pr=np.load(sys.argv[2]); df['P_PyRadiomics']=pr[:,0]*df.scale
ests=[c[2:] for c in df.columns if c.startswith('P_')]
for e in ests: df['FF_'+e]=4*np.pi*df.A/df['P_'+e]**2
df['FF_MATLAB R2023a']=df['FF_skimage/CellProfiler']*(1-0.5/(df['P_skimage/CellProfiler']/df.scale/(2*np.pi)+0.5))**2
ests=ests+['MATLAB R2023a']
key=['src','img','cls','iid']
W={s:df[df.scale==s].set_index(key) for s in [1,2,4]}
common=W[1].index.intersection(W[2].index).intersection(W[4].index); W={s:W[s].loc[common] for s in W}
print("nuclei",len(common))
res={}
for e in ests+['solidity','ecc']:
    c='FF_'+e if e not in ('solidity','ecc') else e
    a=[W[s][c].values for s in [1,2,4]]; sd=a[0].std()
    res[e]=dict(mean1=a[0].mean(),mean2=a[1].mean(),mean4=a[2].mean(),shift2=(a[1]-a[0]).mean()/sd,shift4=(a[2]-a[0]).mean()/sd,
                rhoA=spearmanr(W[1].A,a[0])[0])
print(pd.DataFrame(res).T.round(3).to_string())
if (W[1].reset_index().cls>=0).all():
    R=W[1].reset_index(); keep=(R.cls!=3).values; imgs=R.img.values[keep]; y=R.cls.values[keep]
    base=['A','solidity','ecc','extent','major','minor']
    out={}
    for e in ['none']+ests:
        feats=base+([] if e=='none' else ['FF_'+e])
        X={s:np.c_[np.log(W[s]['A'].values[keep]), W[s][feats[1:]].values[keep]] for s in [1,2,4]}
        accs=[]
        for seed in range(5):
            rng=np.random.default_rng(seed); u=np.unique(imgs); tei=set(rng.choice(u,len(u)//3,replace=False)); te=np.array([i in tei for i in imgs])
            clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,class_weight='balanced')).fit(X[1][~te],y[~te])
            accs.append([balanced_accuracy_score(y[te],clf.predict(X[s][te])) for s in [1,2,4]])
        a=np.array(accs); out[e]={f'{n}x':f"{m:.3f}±{s:.3f}" for n,m,s in zip([40,20,10],a.mean(0),a.std(0))}
        out[e]['drop10x']=f"{(a[:,0]-a[:,2]).mean():.3f}"
    print(pd.DataFrame(out).T.to_string())
