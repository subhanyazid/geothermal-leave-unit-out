"""Physics-based synthetic sensor simulator for geothermal steam turbine units.

Each unit is simulated hourly. Steam properties come from IAPWS-IF97 (via lookup grids).
Three degradation modes (A: corrosion/erosion of blades, B: bearing wear/rotor imbalance,
C: silica scaling at the inlet) plus benign (recovering) excursions and sensor glitches.
Label y=1 if the unit fails within the next H hours.
"""
import numpy as np
import pandas as pd
from scipy.signal import lfilter
from iapws import IAPWS97

H = 72  # prediction horizon (hours)
COND_P_MPA = 0.01  # condenser pressure 0.1 bar

_Pg = np.linspace(3.0, 14.0, 45)
_Ts, _Dh = [], []
for _P in _Pg:
    _s = IAPWS97(P=_P / 10.0, x=1)
    _c = IAPWS97(P=COND_P_MPA, s=_s.s)
    _Ts.append(_s.T - 273.15)
    _Dh.append(_s.h - _c.h)
_Ts = np.array(_Ts)
_Dh = np.array(_Dh)


def tsat(p_bar):
    return np.interp(p_bar, _Pg, _Ts)


def dh_s(p_bar):
    return np.interp(p_bar, _Pg, _Dh)


def _ar1(rng, n, phi, sd):
    sig = sd * np.sqrt(1 - phi ** 2)
    e = rng.normal(0, sig, n + 200)
    return lfilter([1.0], [1.0, -phi], e)[200:]


DEFAULT_SHIFT = dict(noise=1.0, p_gain=1.0, vib_gain=1.0, tb_off=0.0, dur=1.0)


def sim_unit(rng, uid, T=6000, shift=None):
    sh = dict(DEFAULT_SHIFT)
    sh.update(shift or {})
    b = dict(P0=rng.normal(8.0, 0.4), SH0=rng.uniform(2, 8), m0=rng.normal(58, 3),
             eta0=rng.normal(0.78, 0.015), vib0=rng.normal(1.9, 0.25),
             tb0=rng.normal(68, 3), h2s0=rng.normal(120, 25), r0=rng.uniform(0.02, 0.05))
    unit_noise = rng.uniform(0.8, 1.25)
    t = 0
    seg = 0
    frames = []
    while t < T - 400:
        gap = 350 + rng.exponential(900)
        D = rng.uniform(160, 480) * sh['dur']
        onset = t + gap
        tf = onset + D
        if tf >= T - 1:
            seg_end, fail = T, False
        else:
            seg_end, fail = int(tf), True
        n = int(seg_end - t)
        if n < 200:
            break
        k = np.arange(n)
        tt = t + k
        sev = {m: np.zeros(n) for m in 'ABC'}
        mode = rng.choice(list('ABC'), p=[0.35, 0.35, 0.30])
        k_on = int(gap)
        if fail:
            x = np.clip((k - k_on) / D, 0, 1)
            p = rng.uniform(1.3, 2.6)
            sev[mode] += np.where(k >= k_on, x ** p, 0.0)
        n_b = rng.poisson(1.3)
        for _ in range(n_b):
            dur_b = int(rng.uniform(100, 400))
            hi = (k_on - dur_b - 20) if fail else (n - dur_b - 20)
            if hi <= 60:
                continue
            st = int(rng.uniform(60, hi))
            smax = rng.uniform(0.15, 0.55)
            m2 = rng.choice(list('ABC'))
            bump = np.zeros(n)
            bump[st:st + dur_b] = smax * np.sin(np.pi * np.arange(dur_b) / dur_b)
            sev[m2] += bump
        sA, sB, sC = sev['A'], sev['B'], sev['C']
        vA, vB = rng.uniform(1.5, 3.0), rng.uniform(3.5, 5.5)
        tbB, cA = rng.uniform(8, 18), rng.uniform(0.25, 0.45)
        ph1, ph2, ph3 = rng.uniform(0, 2 * np.pi, 3)
        L = (1 + 0.04 * np.sin(2 * np.pi * tt / 24 + ph1) + 0.03 * np.sin(2 * np.pi * tt / 168 + ph2)
             + _ar1(rng, n, 0.97, 0.02))
        for _ in range(rng.poisson(n / 1500)):
            st = int(rng.uniform(0, n - 10))
            du = int(rng.uniform(6, 48))
            L[st:st + du] *= rng.uniform(0.7, 0.9)
        Pwh = b['P0'] * (1 + 0.15 * (L - 1)) + _ar1(rng, n, 0.98, 0.06)
        T_in = tsat(Pwh) + b['SH0'] + _ar1(rng, n, 0.95, 0.4)
        P_in = Pwh * (1 - 0.12 * sC)
        m = b['m0'] * L * (1 - 0.10 * sC)
        eta = b['eta0'] * (1 - 0.10 * sA - 0.04 * sC)
        power = m * dh_s(P_in) * eta / 1000.0
        vib = (b['vib0'] * (0.9 + 0.1 * L) + _ar1(rng, n, 0.9, 0.08) + vA * sA ** 2
               + vB * (np.exp(3 * sB) - 1) / (np.exp(3) - 1) + 0.5 * sC ** 2)
        tb = (b['tb0'] + 6 * (L - 1) + 1.5 * np.sin(2 * np.pi * tt / 24 + ph3)
              + _ar1(rng, n, 0.95, 0.5) + tbB * sB ** 1.5 + 2 * sA)
        h2s = b['h2s0'] + _ar1(rng, n, 0.995, 12)
        ci = 0.03 + b['r0'] * np.cumsum(h2s / 120.0) / 1000.0 + cA * sA + 0.05 * sC
        nz = sh['noise'] * unit_noise
        d = pd.DataFrame(dict(
            unit=uid, seg=seg, t=tt,
            P_in=(P_in + rng.normal(0, 0.03 * nz, n)) * sh['p_gain'],
            T_in=T_in + rng.normal(0, 0.4 * nz, n),
            m_dot=m + rng.normal(0, 0.8 * nz, n),
            power=power + rng.normal(0, 0.25 * nz, n),
            vib=(vib + rng.normal(0, 0.06 * nz, n)) * sh['vib_gain'],
            T_bear=tb + rng.normal(0, 0.4 * nz, n) + sh['tb_off'],
            CI=ci + rng.normal(0, 0.004 * nz, n),
            H2S=h2s + rng.normal(0, 4 * nz, n),
        ))
        g = rng.random(n) < 0.002
        d.loc[g, 'vib'] += rng.uniform(1.0, 3.0, g.sum())
        g = rng.random(n) < 0.001
        d.loc[g, 'P_in'] *= rng.uniform(0.9, 1.1, g.sum())
        if fail:
            ttf = seg_end - tt
            d['ttf'] = ttf
            d['y'] = (ttf <= H).astype(int)
        else:
            d['ttf'] = np.nan
            d['y'] = 0
        d['fail_seg'] = int(fail)
        d['mode'] = mode if fail else ''
        frames.append(d)
        t = seg_end + 48
        seg += 1
    return pd.concat(frames, ignore_index=True)


SIGNALS = ['vib', 'T_bear', 'eta_est', 'CI', 'P_in', 'm_dot', 'superheat', 'power']
RAW = ['P_in', 'T_in', 'm_dot', 'power', 'vib', 'T_bear', 'CI', 'H2S']


def add_features(d):
    """Causal rolling features, computed per segment (maintenance resets history)."""
    out = []
    for (u, s), g in d.groupby(['unit', 'seg'], sort=False):
        g = g.sort_values('t').copy()
        g['superheat'] = g['T_in'] - tsat(g['P_in'])
        g['eta_est'] = g['power'] * 1000.0 / (g['m_dot'] * dh_s(g['P_in']))
        idx = pd.Series(np.arange(len(g), dtype=float), index=g.index)
        w = 72
        var_i = (w ** 2 - 1) / 12.0
        feats = {}
        for sg in SIGNALS:
            x = g[sg]
            feats[f'{sg}_mean24'] = x.rolling(24, min_periods=24).mean()
            feats[f'{sg}_std24'] = x.rolling(24, min_periods=24).std()
            mx = x.rolling(w, min_periods=w).mean()
            mix = (x * idx).rolling(w, min_periods=w).mean()
            mi = idx.rolling(w, min_periods=w).mean()
            feats[f'{sg}_slope72'] = (mix - mi * mx) / var_i * 24.0
            feats[f'{sg}_dev720'] = x - x.rolling(720, min_periods=w).median()
        g = pd.concat([g, pd.DataFrame(feats, index=g.index)], axis=1)
        out.append(g.iloc[w - 1:])
    return pd.concat(out, ignore_index=True)


def feature_columns():
    eng = [f'{s}_{k}' for s in SIGNALS for k in ('mean24', 'std24', 'slope72', 'dev720')]
    return RAW + ['superheat', 'eta_est'] + eng


def build_dataset(seed, n_units=24, T=6000, shift=None, stride=3):
    rng = np.random.default_rng(seed)
    frames = [sim_unit(rng, u, T, shift) for u in range(n_units)]
    d = pd.concat(frames, ignore_index=True)
    d = add_features(d)
    d = d[d['t'] % stride == 0].reset_index(drop=True)
    d['ep'] = np.where(d['fail_seg'] == 1, d['unit'].astype(str) + '_' + d['seg'].astype(str), '')
    return d


def group_of(feat):
    if feat in ('T_in', 'superheat') or feat.startswith('superheat_'):
        return 'superheat/T_in'
    if feat in RAW + ['eta_est']:
        return feat
    return feat.rsplit('_', 1)[0]
