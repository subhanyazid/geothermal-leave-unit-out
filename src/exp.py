"""Run one replicate: python3 exp.py <rep> <outprefix>"""
import sys, time, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_recall_fscore_support,
                             matthews_corrcoef, balanced_accuracy_score, accuracy_score)
from imblearn.over_sampling import SMOTE, ADASYN
from xgboost import XGBClassifier
from sim import build_dataset, feature_columns, RAW, group_of

warnings.filterwarnings('ignore')
FEATS = feature_columns()
REP = int(sys.argv[1])
OUT = sys.argv[2]
STRIDE = 3


def make_model(name, strategy, seed, pos_ratio):
    cw = (strategy == 'class_weight')
    if name == 'RF':
        return RandomForestClassifier(n_estimators=150, max_depth=16, min_samples_leaf=3, n_jobs=1,
                                      random_state=seed, class_weight='balanced_subsample' if cw else None)
    if name == 'XGB':
        return XGBClassifier(n_estimators=250, max_depth=5, learning_rate=0.08, subsample=0.8,
                             colsample_bytree=0.8, tree_method='hist', n_jobs=1, random_state=seed,
                             scale_pos_weight=((1 - pos_ratio) / pos_ratio) if cw else 1.0,
                             eval_metric='logloss', verbosity=0)
    if name == 'LR':
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=500,
                                                                  class_weight='balanced' if cw else None))


def fit_model(name, strategy, Xtr, ytr, seed):
    pr = ytr.mean()
    if strategy == 'SMOTE':
        Xtr, ytr = SMOTE(random_state=seed, k_neighbors=5).fit_resample(Xtr, ytr)
    elif strategy == 'ADASYN':
        Xtr, ytr = ADASYN(random_state=seed, n_neighbors=5).fit_resample(Xtr, ytr)
    m = make_model(name, strategy, seed, pr)
    m.fit(Xtr, ytr)
    return m


def best_threshold(y, p):
    qs = np.unique(np.quantile(p, np.linspace(0.80, 0.999, 120)))
    best, bt = -1, 0.5
    for q in qs:
        pr, rc, f1, _ = precision_recall_fscore_support(y, p >= q, average='binary', zero_division=0)
        if f1 > best:
            best, bt = f1, q
    return bt


def point_metrics(y, p, thr, sfx=''):
    yh = (p >= thr).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y, yh, average='binary', zero_division=0)
    return {f'precision{sfx}': pr, f'recall{sfx}': rc, f'f1{sfx}': f1,
            f'mcc{sfx}': matthews_corrcoef(y, yh), f'bacc{sfx}': balanced_accuracy_score(y, yh),
            f'acc{sfx}': accuracy_score(y, yh)}


def event_metrics(df, p, thr):
    d = df[['unit', 'seg', 't', 'y', 'ttf', 'ep']].copy()
    d['alarm'] = p >= thr
    pos = d[d.y == 1]
    n_ep = pos.ep.nunique()
    det, leads = 0, []
    for ep, g in pos.groupby('ep'):
        a = g[g.alarm]
        if len(a):
            det += 1
            leads.append(a.ttf.max())
    neg = d[(d.y == 0) & d.alarm].sort_values(['unit', 't'])
    n_fa = 0
    if len(neg):
        new = (neg.unit.values[1:] != neg.unit.values[:-1]) | (np.diff(neg.t.values) > 24)
        n_fa = 1 + int(new.sum())
    hours = (d.y == 0).sum() * STRIDE
    return {'event_recall': det / max(n_ep, 1), 'mean_lead_h': float(np.mean(leads)) if leads else 0.0,
            'fa_per_1000h': n_fa / hours * 1000.0, 'n_ep': n_ep}


def score_all(df, p, thr_val):
    y = df.y.values
    r = dict(ap=average_precision_score(y, p), roc=roc_auc_score(y, p), thr=thr_val)
    r.update(point_metrics(y, p, thr_val))
    r.update(point_metrics(y, p, 0.5, '_05'))
    r.update(event_metrics(df, p, thr_val))
    return r


CONFIGS = ([('LR', s, 'full') for s in ('none', 'class_weight', 'SMOTE', 'ADASYN')]
           + [(m, s, 'full') for m in ('RF', 'XGB') for s in ('none', 'class_weight', 'SMOTE', 'ADASYN')]
           + [(m, s, 'raw') for m in ('RF', 'XGB') for s in ('none', 'SMOTE')])
KEEP = [('RF', 'SMOTE'), ('XGB', 'SMOTE'), ('XGB', 'class_weight'), ('RF', 'none'), ('XGB', 'none')]

rows, preds = [], []
t0 = time.time()
rep = REP
d = build_dataset(1000 + rep, n_units=24, stride=STRIDE)
units = np.array(sorted(d.unit.unique()))
rng = np.random.default_rng(rep)
perm = rng.permutation(units)
folds = np.array_split(perm, 4)
for fi, te_units in enumerate(folds):
    pool = np.array([u for u in perm if u not in te_units])
    n_val = len(pool) // 4
    va_units, fit_units = pool[:n_val], pool[n_val:]
    te, va, fit = (d[d.unit.isin(te_units)], d[d.unit.isin(va_units)], d[d.unit.isin(fit_units)])
    for (mn, st, fs) in CONFIGS:
        cols = FEATS if fs == 'full' else RAW
        seed = 7 * rep + fi
        m = fit_model(mn, st, fit[cols].values, fit.y.values, seed)
        pv = m.predict_proba(va[cols].values)[:, 1]
        thr = best_threshold(va.y.values, pv)
        pt = m.predict_proba(te[cols].values)[:, 1]
        r = score_all(te, pt, thr)
        r.update(rep=rep, fold=fi, model=mn, strategy=st, feats=fs)
        rows.append(r)
        if fs == 'full' and (mn, st) in KEEP:
            preds.append(pd.DataFrame(dict(rep=rep, fold=fi, model=mn, strategy=st, y=te.y.values, p=pt,
                                           unit=te.unit.values, t=te.t.values, ttf=te.ttf.values, ep=te.ep.values)))
        if fs == 'full' and (mn, st) in [('RF', 'SMOTE'), ('XGB', 'SMOTE')]:
            base = average_precision_score(te.y.values, pt)
            groups = {}
            for i, c in enumerate(cols):
                groups.setdefault(group_of(c), []).append(i)
            Xte = te[cols].values
            prng = np.random.default_rng(seed)
            for g, idx in groups.items():
                drops = []
                for _ in range(3):
                    Xp = Xte.copy()
                    pi = prng.permutation(len(Xp))
                    Xp[:, idx] = Xp[pi][:, idx]
                    drops.append(base - average_precision_score(te.y.values, m.predict_proba(Xp)[:, 1]))
                rows.append(dict(rep=rep, fold=fi, model=mn, strategy=st, feats='perm', group=g,
                                 perm_drop=float(np.mean(drops))))
    print(f'rep {rep} fold {fi} done {time.time()-t0:.0f}s', flush=True)

rs = np.random.default_rng(100 + rep)
msk = rs.random(len(d)) < 0.7
tr, ts = d[msk], d[~msk]
for (mn, st) in [('RF', 'SMOTE'), ('XGB', 'SMOTE'), ('RF', 'none'), ('XGB', 'none')]:
    m = fit_model(mn, st, tr[FEATS].values, tr.y.values, rep)
    pt = m.predict_proba(ts[FEATS].values)[:, 1]
    r = score_all(ts, pt, 0.5)
    r.update(rep=rep, fold=-1, model=mn, strategy=st, feats='random_split')
    rows.append(r)

pool_units = perm[:19]
vu = perm[19:]
fit, va = d[d.unit.isin(pool_units)], d[d.unit.isin(vu)]
shifts = {'nominal': {}, 'noise x2': dict(noise=2.0), 'sensor bias': dict(p_gain=1.03, vib_gain=1.15, tb_off=3.0),
          'faster degradation': dict(dur=0.6)}
sets = {k: build_dataset(5000 + rep, n_units=12, shift=v, stride=STRIDE) for k, v in shifts.items()}
for (mn, st) in [('RF', 'SMOTE'), ('XGB', 'SMOTE'), ('XGB', 'class_weight'), ('RF', 'none')]:
    m = fit_model(mn, st, fit[FEATS].values, fit.y.values, rep)
    thr = best_threshold(va.y.values, m.predict_proba(va[FEATS].values)[:, 1])
    for k, ds in sets.items():
        pt = m.predict_proba(ds[FEATS].values)[:, 1]
        r = score_all(ds, pt, thr)
        r.update(rep=rep, fold=-2, model=mn, strategy=st, feats='shift', shift=k)
        rows.append(r)

pd.DataFrame(rows).to_csv(f'{OUT}_metrics.csv', index=False)
pd.concat(preds).to_parquet(f'{OUT}_preds.parquet')
if rep == 0:
    d.to_parquet(f'{OUT}_data.parquet')
print('ALL DONE', time.time() - t0, flush=True)
