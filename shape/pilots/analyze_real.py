import numpy as np, pandas as pd, sys
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
df=pd.read_csv(sys.argv[1]); ests=[c[2:] for c in df.columns if c.startswith('P_')]
for e in ests: df['FF_'+e]=4*np.pi*df.A/df['P_'+e]**2
key=['src','img','cls','iid']
W={s:df[df.scale==s].set_index(key) for s in [1,2,4]}
common=W[1].index.intersection(W[2].index).intersection(W[4].index)
W={s:W[s].loc[common] for s in W}
print("nuclei",len(common))
print(f"{'estimator':26s} mean@40x  @20x  @10x | shift/SD 20x 10x | rho(A,FF)@40x")
for e in ests+['solidity']:
    c='FF_'+e if e!='solidity' else e
    a=[W[s][c].values for s in [1,2,4]]; sd=a[0].std()
    print(f"{e:26s} {a[0].mean():.3f} {a[1].mean():.3f} {a[2].mean():.3f} | {(a[1]-a[0]).mean()/sd:+.2f} {(a[2]-a[0]).mean()/sd:+.2f} | {spearmanr(W[1].A,a[0])[0]:+.2f}")
if (W[1].reset_index().cls>=0).all() and len(sys.argv)>2:
    imgs=W[1].reset_index().img.values; y=W[1].reset_index().cls.values
    rng=np.random.default_rng(0); u=np.unique(imgs); te_imgs=set(rng.choice(u,len(u)//3,replace=False)); te=np.array([i in te_imgs for i in imgs])
    base=['A','solidity','ecc','extent','major','minor']
    for e in ['none','skimage/CellProfiler','OpenCV','ImageJ','PyRadiomics','Crofton (skimage)','corrected chain (ours)']:
        if e!='none' and 'FF_'+e not in W[1]: continue
        feats=base+([] if e=='none' else ['FF_'+e])
        X={s:np.c_[np.log(W[s][feats[0]]), W[s][feats[1:]].values] for s in [1,2,4]}
        res=[]
        for clf in [make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight='balanced')), HistGradientBoostingClassifier(class_weight='balanced',random_state=0)]:
            clf.fit(X[1][~te],y[~te]); res+= [balanced_accuracy_score(y[te],clf.predict(X[s][te])) for s in [1,2,4]]
        print(f"{e:24s} LR bacc 40x/20x/10x: {res[0]:.3f} {res[1]:.3f} {res[2]:.3f} | GBT: {res[3]:.3f} {res[4]:.3f} {res[5]:.3f}")
