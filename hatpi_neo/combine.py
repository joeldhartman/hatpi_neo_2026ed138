"""
Per-season combination and the control-position significance.

All fluxes are put on a common zero point ZPref (G = ZPref - 2.5 log10 F).
For every season (and all seasons together), the target flux is compared
with the same quantity measured at the control positions:
  sig = (F_target - mean(F_controls)) / rms(F_controls)
and the magnitude uncertainty is taken from rms(F_controls).
"""

import math

import numpy as np
from astropy.time import Time


def split_seasons(jd, gap=60.0):
    """Split indices into observing seasons separated by gaps > gap days.
    Returns a list of (label, index array); label = year of the first JD."""
    o = np.argsort(jd)
    out = []
    cur = [o[0]]
    for a, b in zip(o[:-1], o[1:]):
        if jd[b] - jd[a] > gap:
            out.append(np.array(cur))
            cur = []
        cur.append(b)
    out.append(np.array(cur))
    return [(str(Time(jd[s].min(), format="jd").datetime.year), s) for s in out]


def weighted_mean(F, E, scale, sel, snrclip=5.0):
    """Inverse-variance weighted mean of F*scale over the selected images,
    excluding images with |F/E| >= snrclip. Returns (mean, n_used)."""
    g = sel & np.isfinite(F) & np.isfinite(E) & (E > 0) & (np.abs(F / E) < snrclip)
    if g.sum() < 3:
        return np.nan, 0
    w = 1 / (E[g] * scale[g]) ** 2
    return np.sum(w * F[g] * scale[g]) / w.sum(), int(g.sum())


def lc_season_stats(F, E, scale, sel, snrclip=5.0):
    """Light-curve-based combination. F, E: (n, npos) arrays, position 0 =
    target, 1.. = controls. Returns (flux, ctl_mean, ctl_rms, n_used)."""
    t, ng = weighted_mean(F[:, 0], E[:, 0], scale, sel, snrclip)
    cm = np.array([weighted_mean(F[:, j], E[:, j], scale, sel, snrclip)[0]
                   for j in range(1, F.shape[1])])
    return t, np.nanmean(cm), np.nanstd(cm), ng


def magnitude(F, zpref):
    return zpref - 2.5 * math.log10(F) if F > 0 else np.nan


def magnitude_with_errors(F, e, zpref):
    """G and its asymmetric errors (faint side +, bright side -) for a flux
    F with uncertainty e."""
    G = magnitude(F, zpref)
    egp = magnitude(F - e, zpref) - G if F - e > 0 else np.inf
    egm = G - magnitude(F + e, zpref) if F + e > 0 else np.nan
    return G, egp, egm
