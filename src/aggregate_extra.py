import json, glob
import numpy as np, pandas as pd
from statsmodels.stats.multitest import multipletests

# ---- Holm correction over all paired tests in tests.json
tests = json.load(open('tests.json'))
names = list(tests); p = [tests[k][1] for k in names]
rej, padj, _, _ = multipletests(p, method='holm')
holm = pd.DataFrame(dict(comparison=names, diff=[tests[k][0] for k in names], p_raw=p, p_holm=padj, wins=[f'{tests[k][2]}/{tests[k][3]}' for k in names]))
holm.to_csv('table_holm.csv', index=False)

ex = pd.concat([pd.read_csv(f) for f in sorted(glob.glob('extra_rep*.csv'))], ignore_index=True)
ex.to_csv('extra_all.csv', index=False)
def ms(g, c):
    return f"{g[c].mean():.3f} ± {g[c].std(ddof=1):.3f}"

# ---- (a) ablation
ab = ex[ex.analysis == 'ablation']
rows = []
for (mn, st, s), g in ab.groupby(['model', 'strategy', 'setting']):
    rows.append(dict(model=mn, strategy=st, features=s,
                     pr_auc=ms(g, 'ap'), f1=ms(g, 'f1'), recall=ms(g, 'recall'), episode_detection=ms(g, 'event_recall')))
pd.DataFrame(rows).to_csv('table_ablation_extra.csv', index=False)

# ---- (b) sensitivity
se = ex[ex.analysis == 'sensitivity']
rows = [dict(model=mn, setting=s, pr_auc=ms(g, 'ap'), f1=ms(g, 'f1'), episode_detection=ms(g, 'event_recall'))
        for (mn, s), g in se.groupby(['model', 'setting'])]
main = pd.read_csv('table_main.csv')
for mn, st in [('RF', 'none'), ('XGB', 'none')]:
    r = main[(main.model == mn) & (main.strategy == st)].iloc[0]
    rows.append(dict(model=mn, setting='depth=16 (main)' if mn == 'RF' else 'depth=5 (main)',
                     pr_auc=f"{r.ap:.3f} ± {r.ap_sd:.3f}", f1=f"{r.f1:.3f} ± {r.f1_sd:.3f}",
                     episode_detection=f"{r.event_recall:.3f}"))
pd.DataFrame(rows).to_csv('table_sensitivity.csv', index=False)

# ---- (c) matched random split vs leave-unit-out (same training size, same threshold procedure)
mr = ex[ex.analysis == 'matched_random']
rows = []
for (mn, st), g in mr.groupby(['model', 'strategy']):
    lo = main[(main.model == mn) & (main.strategy == st)].iloc[0]
    rows.append(dict(model=mn, strategy=st, matched_random_f1=ms(g, 'f1'), matched_random_pr_auc=ms(g, 'ap'),
                     leave_unit_out_f1=f"{lo.f1:.3f} ± {lo.f1_sd:.3f}", leave_unit_out_pr_auc=f"{lo.ap:.3f} ± {lo.ap_sd:.3f}"))
pd.DataFrame(rows).to_csv('table_matched_random.csv', index=False)
print(holm.round(4).to_string())
for f in ['table_ablation_extra.csv', 'table_sensitivity.csv', 'table_matched_random.csv']:
    print('\n==', f); print(pd.read_csv(f).to_string())
