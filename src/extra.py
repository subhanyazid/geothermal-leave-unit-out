"""Extra analyses requested in review (one replicate): python3 extra.py <rep>
 (a) feature ablation: raw / raw+physics (2 derived) / raw+windows on measured signals / full
 (b) hyperparameter sensitivity (RF depth, XGB depth), strategy none
 (c) random row split matched in size to leave-unit-out folds, same threshold procedure
Uses the same unit folds, seeds and CONFIG as exp.py."""
import sys, time, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sim import build_dataset, feature_columns, RAW

warnings.filterwarnings('ignore')
REP = int(sys.argv[1])
FEATS = feature_columns()
MEAS6 = ['vib', 'T_bear', 'CI', 'P_in', 'm_dot', 'power']
WIN6 = [f'{s}_{k}' for s in MEAS6 for k in ('mean24', 'std24', 'slope72', 'dev720')]
SETS = {'raw': RAW, 'phys': RAW + ['superheat', 'eta_est'], 'winraw': RAW + WIN6, 'full': FEATS}
assert set(SETS['full']) == set(FEATS) and len(SETS['winraw']) == 32 and len(SETS['phys']) == 10


def fit(name, strategy, X, y, seed, depth=16, xdepth=5):
    if strategy == 'SMOTE':
        X, y = SMOTE(random_state=seed, k_neighbors=5).fit_resample(X, y)
    if name == 'RF':
        m = RandomForestClassifier(n_estimators=150, max_depth=depth, min_samples_leaf=3, n_jobs=1, random_state=seed)
    else:
        m = XGBClassifier(n_estimators=250, max_depth=xdepth, learning_rate=0.08, subsample=0.8,
                          colsample_bytree=0.8, tree_method='hist', n_jobs=1, random_state=seed,
                          eval_metric='logloss', verbosity=0)
    return m.fit(X, y)


def best_threshold(y, p):
    qs = np.unique(np.quantile(p, np.linspace(0.80, 0.999, 120)))
    best, bt = -1, 0.5
    for q in qs:
        _, _, f1, _ = precision_recall_fscore_support(y, p >= q, average='binary', zero_division=0)
        if f1 > best:
            best, bt = f1, q
    return bt


def event_recall(df, p, thr):
    d = df[['y', 'ep']].copy(); d['alarm'] = p >= thr
    pos = d[d.y == 1]
    n = pos.ep.nunique()
    det = sum(g.alarm.any() for _, g in pos.groupby('ep'))
    return det / max(n, 1)


def score(df, p, thr):
    y = df.y.values
    _, _, f1, _ = precision_recall_fscore_support(y, p >= thr, average='binary', zero_division=0)
    pr, rc, _, _ = precision_recall_fscore_support(y, p >= thr, average='binary', zero_division=0)
    return dict(ap=average_precision_score(y, p), f1=f1, precision=pr, recall=rc,
                event_recall=event_recall(df, p, thr))


t0 = time.time()
rows = []
d = build_dataset(1000 + REP, n_units=24, stride=3)
units = np.array(sorted(d.unit.unique()))
perm = np.random.default_rng(REP).permutation(units)
folds = np.array_split(perm, 4)
for fi, te_units in enumerate(folds):
    pool = np.array([u for u in perm if u not in te_units])
    n_val = len(pool) // 4
    va_units, fit_units = pool[:n_val], pool[n_val:]
    te, va, tr = d[d.unit.isin(te_units)], d[d.unit.isin(va_units)], d[d.unit.isin(fit_units)]
    seed = 7 * REP + fi

    # (a) ablation, same protocol as main experiment
    for fs, cols in SETS.items():
        for mn in ('RF', 'XGB'):
            for st in ('none', 'SMOTE'):
                m = fit(mn, st, tr[cols].values, tr.y.values, seed)
                thr = best_threshold(va.y.values, m.predict_proba(va[cols].values)[:, 1])
                pt = m.predict_proba(te[cols].values)[:, 1]
                rows.append(dict(analysis='ablation', rep=REP, fold=fi, model=mn, strategy=st, setting=fs, **score(te, pt, thr)))

    # (b) hyperparameter sensitivity (strategy none, full features)
    for mn, kw in [('RF', dict(depth=8)), ('RF', dict(depth=0)), ('XGB', dict(xdepth=3)), ('XGB', dict(xdepth=7))]:
        kw2 = {k: (None if (k == 'depth' and v == 0) else v) for k, v in kw.items()}
        m = fit(mn, 'none', tr[FEATS].values, tr.y.values, seed, **kw2)
        thr = best_threshold(va.y.values, m.predict_proba(va[FEATS].values)[:, 1])
        pt = m.predict_proba(te[FEATS].values)[:, 1]
        setting = f"depth={kw2.get('depth', kw2.get('xdepth'))}"
        rows.append(dict(analysis='sensitivity', rep=REP, fold=fi, model=mn, strategy='none', setting=setting, **score(te, pt, thr)))

    # (c) random row split with the same numbers of train / validation / test rows as this fold
    n_tr, n_va, n_te = len(tr), len(va), len(te)
    idx = np.random.default_rng(100 * REP + fi).permutation(len(d))
    i_tr, i_va, i_te = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:n_tr + n_va + n_te]
    Rtr, Rva, Rte = d.iloc[i_tr], d.iloc[i_va], d.iloc[i_te]
    for mn in ('RF', 'XGB'):
        for st in ('none', 'SMOTE'):
            m = fit(mn, st, Rtr[FEATS].values, Rtr.y.values, seed)
            thr = best_threshold(Rva.y.values, m.predict_proba(Rva[FEATS].values)[:, 1])
            pt = m.predict_proba(Rte[FEATS].values)[:, 1]
            _, _, f1, _ = precision_recall_fscore_support(Rte.y.values, pt >= thr, average='binary', zero_division=0)
            rows.append(dict(analysis='matched_random', rep=REP, fold=fi, model=mn, strategy=st, setting='random_rows',
                             ap=average_precision_score(Rte.y.values, pt), f1=f1))
    print(f'rep {REP} fold {fi} done {time.time()-t0:.0f}s', flush=True)

pd.DataFrame(rows).to_csv(f'extra_rep{REP}.csv', index=False)
print('ALL DONE', round(time.time() - t0), flush=True)
