import os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.stats import wilcoxon
from sklearn.metrics import precision_recall_curve, average_precision_score
import sim

os.makedirs('fig', exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
                     'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'figure.dpi': 150,
                     'axes.spines.top': False, 'axes.spines.right': False})
m = pd.concat([pd.read_csv(f'rep{r}_metrics.csv') for r in range(3)], ignore_index=True)
m.to_csv('results_metrics.csv', index=False)
P = pd.concat([pd.read_parquet(f'rep{r}_preds.parquet') for r in range(3)], ignore_index=True)
cv = m[m.feats.isin(['full', 'raw'])].copy()
main = cv[cv.feats == 'full']
res = {}

cols = ['ap', 'roc', 'precision', 'recall', 'f1', 'mcc', 'event_recall', 'mean_lead_h', 'fa_per_1000h']
order = [('LR', 'none'), ('LR', 'class_weight'), ('LR', 'SMOTE'), ('LR', 'ADASYN'), ('RF', 'none'), ('RF', 'class_weight'), ('RF', 'SMOTE'),
         ('RF', 'ADASYN'), ('XGB', 'none'), ('XGB', 'class_weight'), ('XGB', 'SMOTE'), ('XGB', 'ADASYN')]
rows = []
for (mn, st) in order:
    g = main[(main.model == mn) & (main.strategy == st)]
    rows.append([mn, st] + [g[c].mean() for c in cols] + [g[c].std(ddof=1) for c in cols])
tab = pd.DataFrame(rows, columns=['model', 'strategy'] + cols + [c + '_sd' for c in cols])
tab.to_csv('table_main.csv', index=False)

abl = []
for mn in ('RF', 'XGB'):
    for st in ('none', 'SMOTE'):
        for fs in ('raw', 'full'):
            g = cv[(cv.model == mn) & (cv.strategy == st) & (cv.feats == fs)]
            abl.append([mn, st, fs] + [g[c].mean() for c in ('ap', 'f1', 'recall', 'event_recall')] +
                       [g[c].std(ddof=1) for c in ('ap', 'f1', 'recall', 'event_recall')])
abl = pd.DataFrame(abl, columns=['model', 'strategy', 'feats', 'ap', 'f1', 'recall', 'event_recall',
                                 'ap_sd', 'f1_sd', 'recall_sd', 'event_recall_sd'])
abl.to_csv('table_ablation.csv', index=False)


def paired(a, b, metric):
    A = main[(main.model == a[0]) & (main.strategy == a[1])].sort_values(['rep', 'fold'])[metric].values
    B = main[(main.model == b[0]) & (main.strategy == b[1])].sort_values(['rep', 'fold'])[metric].values
    try:
        p = wilcoxon(A, B).pvalue
    except ValueError:
        p = 1.0
    return [float(np.mean(A - B)), float(p), int((A > B).sum()), len(A)]


tests = {}
for a, b in [(('RF', 'SMOTE'), ('RF', 'none')), (('XGB', 'SMOTE'), ('XGB', 'none')),
             (('RF', 'SMOTE'), ('RF', 'class_weight')), (('XGB', 'SMOTE'), ('XGB', 'class_weight')),
             (('XGB', 'none'), ('RF', 'none')), (('XGB', 'SMOTE'), ('RF', 'SMOTE')),
             (('XGB', 'class_weight'), ('XGB', 'none')), (('RF', 'class_weight'), ('RF', 'none')),
             (('RF', 'none'), ('LR', 'none')), (('XGB', 'none'), ('LR', 'none'))]:
    for metric in ('ap', 'f1'):
        tests[f'{a[0]}-{a[1]} vs {b[0]}-{b[1]} [{metric}]'] = paired(a, b, metric)
json.dump(tests, open('tests.json', 'w'), indent=1)

rs = m[m.feats == 'random_split']
leak = []
for mn, st in [('RF', 'SMOTE'), ('XGB', 'SMOTE'), ('RF', 'none'), ('XGB', 'none')]:
    a = rs[(rs.model == mn) & (rs.strategy == st)]
    b = main[(main.model == mn) & (main.strategy == st)]
    leak.append([mn, st] + [a[c].mean() for c in ('acc_05', 'precision_05', 'recall_05', 'f1_05', 'ap')]
                + [b[c].mean() for c in ('acc_05', 'precision_05', 'recall_05', 'f1_05', 'ap')]
                + [a['f1_05'].std(ddof=1), b['f1_05'].std(ddof=1)])
leak = pd.DataFrame(leak, columns=['model', 'strategy', 'rs_acc', 'rs_prec', 'rs_rec', 'rs_f1', 'rs_ap',
                                   'lo_acc', 'lo_prec', 'lo_rec', 'lo_f1', 'lo_ap', 'rs_f1_sd', 'lo_f1_sd'])
leak.to_csv('table_leak.csv', index=False)

sh = m[m.feats == 'shift']
shift = sh.groupby(['model', 'strategy', 'shift'])[['ap', 'recall', 'precision', 'event_recall', 'fa_per_1000h']].mean().reset_index()
shift.to_csv('table_shift.csv', index=False)

pm = m[m.feats == 'perm']
pi = pm.groupby(['model', 'group']).perm_drop.agg(['mean', 'std']).reset_index()
pi.to_csv('table_perm.csv', index=False)

# ---------------- figures ----------------
rng = np.random.default_rng(42)
x = None
for u in range(60):
    xx = sim.sim_unit(rng, u, T=6000)
    ff = xx[xx.fail_seg == 1]
    if len(ff) and ff['mode'].eq('B').any():
        x = xx
        break
seg_f = x[(x.fail_seg == 1) & (x['mode'] == 'B')].seg.iloc[0]
s = x[x.seg == seg_f].copy()
s['rel'] = s.t - s.t.iloc[0]
fig, axes = plt.subplots(5, 1, figsize=(6.2, 5.6), sharex=True)
sigs = [('P_in', 'P$_{in}$ (bar)'), ('power', 'P$_{el}$ (MW)'), ('vib', 'Vibr. (mm/s)'),
        ('T_bear', 'T$_{bear}$ (°C)'), ('CI', 'CI (–)')]
for ax, (c, lab) in zip(axes, sigs):
    ax.plot(s.rel, s[c], lw=0.6, color='#1f4e79')
    ax.set_ylabel(lab)
    ax.axvspan(s.rel.max() - 71, s.rel.max() + 1, color='#d62728', alpha=0.15, lw=0)
axes[-1].set_xlabel('Operating hours since last overhaul')
axes[0].text(s.rel.max() - 70, axes[0].get_ylim()[1], ' positive label\n (last 72 h)', va='top', fontsize=6.5, color='#a01010')
plt.tight_layout()
plt.savefig('fig/fig1_signals.png', dpi=300)
plt.close()
res['fig1_mode'] = 'B'

fig, ax = plt.subplots(figsize=(6.2, 1.9))
ax.set_xlim(0, 100)
ax.set_ylim(0, 30)
ax.axis('off')
boxes = ['Physics-based\nsimulator\n(IAPWS-IF97,\n3 failure modes)', 'Causal feature\nextraction\n(24 h / 72 h /\n720 h windows)',
         'Leave-unit-out\nsplit\n(fit / validation /\ntest units)', 'Imbalance\nhandling (none,\nclass weight,\nSMOTE, ADASYN)',
         'Classifiers\nLR, RF, XGBoost;\nthreshold set on\nvalidation units']
for i, txt in enumerate(boxes):
    x0 = 1.0 + i * 19.7
    ax.add_patch(FancyBboxPatch((x0, 6), 16.2, 20, boxstyle='round,pad=0.3,rounding_size=1.0', fc='#eaf1f8', ec='#1f4e79', lw=0.8))
    ax.text(x0 + 8.1, 16, txt, ha='center', va='center', fontsize=5.8)
    if i < 4:
        ax.add_patch(FancyArrowPatch((x0 + 16.8, 16), (x0 + 19.4, 16), arrowstyle='-|>', mutation_scale=7, color='#1f4e79'))
ax.text(50, 1.5, 'Evaluation on held-out units: PR-AUC, F1, MCC, event-level detection, false alarms, lead time',
        ha='center', fontsize=6.5, style='italic')
plt.savefig('fig/fig2_pipeline.png', dpi=300, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(3.2, 2.8))
cmap = {('RF', 'SMOTE'): '#d62728', ('RF', 'none'): '#ff9896', ('XGB', 'SMOTE'): '#1f77b4',
        ('XGB', 'class_weight'): '#2ca02c', ('XGB', 'none'): '#aec7e8'}
for (mn, st), col in cmap.items():
    g = P[(P.model == mn) & (P.strategy == st) & (P.rep == 0)]
    pr, rc, _ = precision_recall_curve(g.y, g.p)
    ax.plot(rc, pr, color=col, lw=1.1, label=f'{mn}-{st} (AP={average_precision_score(g.y, g.p):.2f})')
ax.axhline(P[P.rep == 0].y.mean(), color='gray', ls=':', lw=0.8, label='prevalence')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.legend(loc='upper right', frameon=False, fontsize=6)
ax.set_ylim(0, 1.02)
plt.tight_layout()
plt.savefig('fig/fig3_pr.png', dpi=300)
plt.close()

fig, ax = plt.subplots(figsize=(3.2, 2.8))
lab = [f'{r.model}\n{r.strategy.replace("class_weight", "c.w.")}' for r in tab.itertuples()]
cols_ = ['#7f7f7f' if r.model == 'LR' else ('#d62728' if r.model == 'RF' else '#1f77b4') for r in tab.itertuples()]
ax.bar(range(len(tab)), tab.ap, yerr=tab.ap_sd, color=cols_, capsize=1.5, error_kw={'lw': 0.6})
ax.set_xticks(range(len(tab)))
ax.set_xticklabels(lab, fontsize=5, rotation=90)
ax.set_ylabel('PR-AUC (mean ± sd)')
plt.tight_layout()
plt.savefig('fig/fig4_ap.png', dpi=300)
plt.close()

fig, axs = plt.subplots(1, 2, figsize=(6.2, 2.4), sharey=True)
for ax, mn in zip(axs, ('RF', 'XGB')):
    g = pi[pi.model == mn].sort_values('mean')
    ax.barh(g.group, g['mean'], xerr=g['std'], color='#d62728' if mn == 'RF' else '#1f77b4', capsize=1.5, error_kw={'lw': 0.6})
    ax.set_title(f'{mn} + SMOTE')
    ax.set_xlabel('Drop in PR-AUC when sensor group permuted')
plt.tight_layout()
plt.savefig('fig/fig5_perm.png', dpi=300)
plt.close()

fig, ax = plt.subplots(figsize=(3.2, 2.6))
w = 0.35
xi = np.arange(len(leak))
ax.bar(xi - w / 2, leak.rs_f1, w, label='Random row split', color='#ff7f0e')
ax.bar(xi + w / 2, leak.lo_f1, w, label='Leave-unit-out', color='#1f4e79')
ax.set_xticks(xi)
ax.set_xticklabels([f'{a}\n{b}' for a, b in zip(leak.model, leak.strategy)])
ax.set_ylabel('F1 at threshold 0.5')
ax.set_ylim(0, 1.05)
ax.legend(frameon=False, fontsize=6, loc='lower right')
plt.tight_layout()
plt.savefig('fig/fig6_leak.png', dpi=300)
plt.close()

fig, ax = plt.subplots(figsize=(3.2, 2.6))
shifts = ['nominal', 'noise x2', 'sensor bias', 'faster degradation']
cfgs = [('RF', 'SMOTE'), ('RF', 'none'), ('XGB', 'SMOTE'), ('XGB', 'class_weight')]
w = 0.2
for i, (mn, st) in enumerate(cfgs):
    v = [shift[(shift.model == mn) & (shift.strategy == st) & (shift['shift'] == k)].ap.values[0] for k in shifts]
    ax.bar(np.arange(4) + (i - 1.5) * w, v, w, label=f'{mn}-{st}')
ax.set_xticks(range(4))
ax.set_xticklabels(['nominal', 'noise ×2', 'sensor\nbias', 'faster\ndegr.'])
ax.set_ylabel('PR-AUC')
ax.legend(frameon=False, fontsize=5.5)
plt.tight_layout()
plt.savefig('fig/fig7_shift.png', dpi=300)
plt.close()

from PIL import Image
figs = {}
for k, f in [('f1', 'fig1_signals'), ('f2', 'fig2_pipeline'), ('f3', 'fig3_pr'), ('f4', 'fig4_ap'),
             ('f5', 'fig5_perm'), ('f6', 'fig6_leak'), ('f7', 'fig7_shift')]:
    im = Image.open(f'fig/{f}.png')
    figs[k] = dict(path=f'fig/{f}.png', aspect=im.size[1] / im.size[0])
figs['f1']['mode'] = 'B'
json.dump(figs, open('figs.json', 'w'))

ds = []
for rep in range(3):
    d = sim.build_dataset(1000 + rep, n_units=24, stride=3)
    ds.append(dict(rep=rep, rows=int(len(d)), pos=int(d.y.sum()), rate=float(d.y.mean()),
                   episodes=int(d.ep.replace('', np.nan).nunique()),
                   modes={k: int(v) for k, v in d[d.fail_seg == 1].groupby('mode').ep.nunique().items()}))
json.dump(ds, open('ds_stats.json', 'w'), indent=1)

pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 30)
print(tab.round(3).to_string())
print(abl.round(3).to_string())
for k, v in tests.items():
    print(k, [round(z, 4) for z in v])
print(leak.round(3).to_string())
print(shift.round(3).to_string())
print(pi.round(4).to_string())
print(json.dumps(ds))
