"""Analysis of RIDER coffee-break results for both papers."""
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

pd.set_option('display.width', 220)


def icc31(Y):
    """ICC(3,1), consistency, two-way mixed; Y: n_subjects x k_raters."""
    Y = np.asarray(Y, float); n, k = Y.shape
    gm = Y.mean(); ms_r = k * ((Y.mean(1) - gm) ** 2).sum() / (n - 1)
    ms_e = ((Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + gm) ** 2).sum() / ((n - 1) * (k - 1))
    return (ms_r - ms_e) / (ms_r + (k - 1) * ms_e)


def ccc(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return 2 * np.cov(a, b, bias=True)[0, 1] / (a.var() + b.var() + (a.mean() - b.mean()) ** 2)


IDX = pd.read_csv('index.csv')
UNIFORM = set(IDX.series[IDX.uniform]) if 'uniform' in IDX else set(IDX.series)
print('QC: uniform series', len(UNIFORM), 'of', len(IDX))

# ---------------- paper 1: shape ----------------
S = pd.read_csv('rider_shape.csv')
S = S[S.series.isin(UNIFORM)].drop(columns=['visit']).merge(IDX[['series', 'visit']], on='series')
TWO = set(IDX.groupby('patient').visit.max().pipe(lambda v: v[v == 2]).index)   # patients with exactly test + retest
print('=== SHAPE: series', len(S), 'patients', S.patient.nunique())
for est in ['prad', 'crofton', 'rcc']:
    print(f'-- {est}: mean by setting')
    print(S.groupby(['kernel', 'thickness'])[est].agg(['mean', 'std', 'count']).round(3).T.to_string())
W = S.pivot_table(index=['patient', 'visit'], columns=['kernel', 'thickness'], values=['prad', 'crofton', 'rcc'])
for kern in ['LUNG', 'STANDARD']:
    for t in [2.5, 5.0]:
        line = []
        for est in ['prad', 'crofton', 'rcc']:
            d = (W[(est, kern, t)] - W[(est, kern, 1.25)]).dropna()
            if len(d) < 5:
                continue
            p = wilcoxon(d).pvalue
            line.append(f"{est}: {d.mean():+.3f} [{d.mean()-1.96*d.sem():+.3f},{d.mean()+1.96*d.sem():+.3f}] p={p:.1e} n={len(d)}")
        print(f'{kern} 1.25->{t}: ' + ' | '.join(line))
print('-- concordance (CCC) between 1.25 and 5 mm, per kernel')
for kern in ['LUNG', 'STANDARD']:
    print(kern, {est: round(ccc(*W[[(est, kern, 1.25), (est, kern, 5.0)]].dropna().values.T), 3) for est in ['prad', 'crofton', 'rcc']})
print('-- test-retest ICC(3,1) per setting')
T = S[S.patient.isin(TWO)].pivot_table(index=['patient', 'kernel', 'thickness'], columns='visit', values=['prad', 'crofton', 'rcc']).dropna()
for (kern, t), g in T.groupby(level=['kernel', 'thickness']):
    print(kern, t, {est: round(icc31(g[est].values), 3) for est in ['prad', 'crofton', 'rcc']}, 'n', len(g))

# ---------------- paper 2: texture ----------------
X = pd.read_csv('rider_texture.csv')
X['series'] = X.roi.str.split('_').str[0]
X = X[X.series.isin(UNIFORM)].drop(columns=['visit']).merge(IDX[['series', 'visit']], on='series')
s = X[X.part == 'soft'].copy()
s['neff'] = s.N / s.xi ** 3
P = s.pivot_table(index=['roi', 'kernel', 'thickness', 'feature', 'neff', 'ref', 'xi'], columns='method', values='value').reset_index()
P['gated'] = np.where(P.neff >= 20, P.JK, P.plugin)
print('=== TEXTURE soft tissue: ROIs', P.roi.nunique())
rows = []
for (f, k, t), g in P.groupby(['feature', 'kernel', 'thickness']):
    e = {m: g[m] - g.ref for m in ['plugin', 'JK', 'gated']}
    rows.append(dict(feature=f, kernel=k, thickness=t, xi=g.xi.median(), applied=(g.neff >= 20).mean(), n=g.roi.nunique(),
                     **{f'bias_{m}': e[m].mean() for m in e}, **{f'rmse_{m}': np.sqrt((e[m] ** 2).mean()) for m in e}))
R = pd.DataFrame(rows); R.to_csv('rider_soft_summary.csv', index=False)
print(R[R.feature.isin(['JointEntropy', 'Entropy', 'Correlation', 'Contrast'])].round(3).to_string())
tm = X[X.part == 'tumour']
Q = tm.pivot_table(index=['patient', 'visit', 'kernel', 'thickness', 'feature', 'N', 'xi'], columns='method', values='value').reset_index()
Q['neff'] = Q.N / Q.xi ** 3
Q['gated'] = np.where(Q.neff >= 20, Q.JK, Q.plugin)
Q.to_csv('rider_tumour_texture.csv', index=False)
print('=== TEXTURE tumours: share gated-applied', (Q.neff >= 20).mean().round(2))
for f in ['JointEntropy', 'Entropy', 'Correlation']:
    q = Q[Q.feature == f].pivot_table(index=['patient', 'visit'], columns=['kernel', 'thickness'], values=['plugin', 'gated'])
    for kern in ['LUNG', 'STANDARD']:
        out = []
        for m in ['plugin', 'gated']:
            d = (q[(m, kern, 5.0)] - q[(m, kern, 1.25)]).dropna()
            out.append(f"{m}: 1.25->5 shift {d.mean():+.3f} (sd {d.std():.3f}, n={len(d)})")
        print(f, kern, ' | '.join(out))
    rt = Q[(Q.feature == f) & Q.patient.isin(TWO)].pivot_table(index=['patient', 'kernel', 'thickness'], columns='visit', values=['plugin', 'gated']).dropna()
    print(f, 'test-retest ICC', {m: round(icc31(rt[m].values), 3) for m in ['plugin', 'gated']})
