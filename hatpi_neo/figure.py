"""
Summary figure: stacked stamps per season and for all seasons (top), the
sky path with the observed segments of each season (middle), and the
per-season G magnitudes of both methods with the Horizons V prediction
(bottom).
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, ConnectionPatch

COLS = ["#00796b", "#558b2f", "#9e9d24", "#ef6c00", "#6a1b9a", "#c62828"]


def julian_epoch(jd):
    return 2000.0 + (np.asarray(jd) - 2451545.0) / 365.25


def make_figure(outbase, seasons, jd, stamps, results, hjd, hra, hdec, hmag,
                lc_method, img_method, lc_label="Light Curve-based stacking",
                img_label="Image-based stacking", half=10, marker_radius=2.0):
    """seasons: [(label, index array)] (light-curve seasons, all images);
    stamps: {label or 'all': (image, centre pixel H, sig, n)} from the image
    stacks; results: {(method, label): dict(G=, egp=, egm=, jdmin=, jdmax=)}."""
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "xtick.labelsize": 10,
                         "ytick.labelsize": 10, "xtick.direction": "in",
                         "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
                         "axes.linewidth": 0.8, "font.family": "serif",
                         "mathtext.fontset": "dejavuserif"})
    labels = [l for l, _ in seasons]
    nseas = len(seasons)
    fig = plt.figure(figsize=(10, 11))
    gs = fig.add_gridspec(3, nseas + 2, height_ratios=[0.9, 1.35, 1.0], hspace=0.22,
                          wspace=0.25)

    # stamps, one colour scale for all
    crops = []
    for lab in labels + ["all"]:
        img, H, sig, ng = stamps[lab]
        crops.append((img[H - half:H + half + 1, H - half:H + half + 1], lab, sig, ng))
    vmax = np.nanpercentile(np.abs(np.concatenate([c[0].ravel() for c in crops])), 98)
    stamp_axes = []
    for k, (im, lab, sig, ng) in enumerate(crops):
        ax = fig.add_subplot(gs[0, nseas:nseas + 2] if lab == "all" else gs[0, k])
        mappable = ax.imshow(im, origin="lower", cmap="magma", vmin=-0.5 * vmax, vmax=vmax,
                             interpolation="nearest")
        ax.add_patch(Circle((half, half), marker_radius, fill=False, ec="cyan", lw=1.2))
        ax.set_xticks([])
        ax.set_yticks([])
        c = COLS[-1] if lab == "all" else COLS[k]
        for sp in ax.spines.values():
            sp.set_color(c)
            sp.set_linewidth(2.5)
        ax.set_title("%s  (N=%d)\n%.1f$\\sigma$" % ("All seasons" if lab == "all" else lab,
                                                   ng, sig), fontsize=10)
        stamp_axes.append(ax)
    cb = fig.colorbar(mappable, ax=stamp_axes[-1], fraction=0.05, pad=0.03)
    cb.set_label("Scaled ADU", fontsize=10)
    cb.ax.tick_params(labelsize=9, direction="out", right=True)

    # sky path
    axp = fig.add_subplot(gs[1, :])
    axp.plot(hra, hdec, color="0.6", lw=1)
    for k, (lab, s) in enumerate(seasons):
        ra = np.interp(jd[s], hjd, hra)
        de = np.interp(jd[s], hjd, hdec)
        axp.plot(ra, de, ".", color=COLS[k], ms=4)
        i0 = np.argmin(np.abs(jd[s] - np.median(jd[s])))
        fig.add_artist(ConnectionPatch(xyA=(half, 0), coordsA=stamp_axes[k].transData,
                                       xyB=(ra[i0], de[i0]), coordsB=axp.transData,
                                       color=COLS[k], lw=1))
    axp.set_xlabel("RA (deg)")
    axp.set_ylabel("Dec (deg)")

    # per-season magnitudes
    axl = fig.add_subplot(gs[2, :])
    tt = np.linspace(hjd.min(), hjd.max(), 2000)
    axl.plot(julian_epoch(tt), np.interp(tt, hjd, hmag), color="0.5", lw=1)
    handles = []
    for meth, m_, dx, lab_ in [(lc_method, "o", -0.05, lc_label),
                               (img_method, "s", +0.05, img_label)]:
        for si, lab in enumerate(labels):
            r = results.get((meth, lab))
            if not r or not np.isfinite(r["G"]):
                continue
            axl.errorbar(julian_epoch(0.5 * (r["jdmin"] + r["jdmax"])) + dx, r["G"],
                         yerr=[[r["egm"]], [min(r["egp"], 2)]], fmt=m_, color=COLS[si],
                         ms=6, mec="k", mew=0.6, capsize=2.5, elinewidth=1.2)
        handles.append(Line2D([], [], ls="none", marker=m_, color="0.6", mec="k", mew=0.6,
                              ms=6, label=lab_))
    handles.append(Line2D([], [], color="0.5", lw=1, label="Horizons V prediction"))
    axl.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.56, 0.0), fontsize=9,
               frameon=False)
    axl.set_xlabel("Julian Year Number")
    axl.set_ylabel("G (mag)")
    axl.set_ylim(21.2, 18.8)
    fig.savefig(outbase + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(outbase + ".pdf", bbox_inches="tight")
    plt.close(fig)
