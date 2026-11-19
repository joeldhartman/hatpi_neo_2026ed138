"""
run_analysis(): the full processing used for the RNAAS note on 2026 ED138.

Inputs: a JPL Horizons observer table and an image list (see inputs.py).
Steps:
  1. predicted target position in every image (Horizons + WCS);
  2. "Light Curve-based stacking": forced fiphot photometry at the target
     and control positions with the static residual correction
     (lcphot.py); per-image error = max(eflux_corr, eflux_emp); images
     with |S/N| >= 5 at a position are excluded at that position; inverse-
     variance weighted mean per season;
  3. "Image-based stacking": static-residual-corrected, shifted cutouts
     (imstack.py) of the images whose uncorrected photometry has
     |S/N| <= 5 and no masked pixel in the aperture, combined per season
     with an inverse-variance weighted, sigma-clipped mean, and measured
     with the same aperture;
  4. for both, the significance and magnitude errors from the control
     positions (combine.py), a per-season table and the figure.
"""

import os
import sys
import time

import numpy as np
from astropy.io import fits

from . import combine, figure, imstack, inputs, lcphot

LC_METHOD = "light-curve stacking (residual-corrected, empirical error floor)"
IMG_METHOD = "image stacking (residual-corrected)"


def _fmt(v, f="%.4f"):
    return "nan" if not np.isfinite(v) else f % v


def run_analysis(horizons_file, image_list, outdir,
                 fiphot="fiphot",
                 aperture="1.45:8:8",
                 ncontrol=10, control_radii=(50.0, 90.0),
                 track_mask_radius=8.0, nmin=5, clip=4.0,
                 snr_clip=5.0, pix_clip=4.0, max_mask_frac=0.0,
                 half_size=40, xy_offset=0.5,
                 resample_fluxcorr=0.901, nres_check=5,
                 season_gap=60.0, nthreads=8, make_plot=True):
    """Run the analysis; results are written to outdir.

    Parameters (defaults are the values used in the note)
    ----------
    horizons_file : JPL Horizons observer table (RA/Dec and APmag).
    image_list : image list file (see hatpi_neo.inputs).
    fiphot : fiphot executable (fitsh/HATpipe, 0.9zc or later).
    aperture : fiphot aperture "r:annulus_inner:annulus_width" (pixels).
    ncontrol, control_radii : control positions per ring and ring radii (px).
    track_mask_radius : images whose target lies within this many pixels of
        a position do not contribute to the static residual there.
    nmin : minimum number of images for the static residual.
    clip : clipping (MAD sigma) for the light-curve residual estimate.
    snr_clip : images with |S/N| above this at a position are excluded.
    pix_clip : per-pixel clipping (sigma) in the image stacks.
    max_mask_frac : maximum masked fraction of the target aperture.
    half_size : cutout half size (pixels).
    xy_offset : X,Y (as passed to fiphot) minus the 0-based pixel-centre
        coordinate; 0.5 for the fitsh convention.
    resample_fluxcorr : fraction of a point source's aperture flux kept
        after the bilinear resampling; None to measure it on the 'calimage'
        files of the image list (nres_check of them).
    season_gap : gap (days) separating observing seasons.
    nthreads : parallel fiphot processes.

    Returns a dict with the per-season results.
    """
    t0 = time.time()
    os.makedirs(outdir, exist_ok=True)
    workdir = os.path.join(outdir, "work")
    os.makedirs(workdir, exist_ok=True)
    log = open(os.path.join(outdir, "run_log.txt"), "w")

    def say(s=""):
        print(s, file=sys.stderr)
        log.write(s + "\n")
        log.flush()

    r_ap, rin, rwid = (float(v) for v in aperture.split(":"))
    H = int(half_size)
    S = 2 * H + 1
    fiphot = lcphot.check_fiphot(fiphot)

    hjd, hra, hdec, hmag = inputs.read_horizons(horizons_file)
    tab = inputs.read_image_list(image_list)
    n = len(tab["jd"])
    jd, zp = tab["jd"], tab["zp"]
    ra, dec, X, Y = inputs.target_positions(tab, hjd, hra, hdec)
    offsets = inputs.control_offsets(ncontrol, control_radii)
    zpref = float(np.nanmedian(zp))
    scale = 10 ** (0.4 * (zpref - zp))
    say("# %d images, JD %.2f - %.2f; ZPref = %.4f; %d control positions"
        % (n, jd.min(), jd.max(), zpref, len(offsets)))

    # ---- light-curve photometry ----------------------------------------------
    P = lcphot.residual_corrected_photometry(
        tab, X, Y, offsets, os.path.join(workdir, "photometry"), fiphot,
        aperture=aperture, track_mask_radius=track_mask_radius, nmin=nmin, clip=clip,
        nthreads=nthreads)
    E3 = np.fmax(P["ecorr"], P["eemp"])

    with open(os.path.join(outdir, "lightcurve.txt"), "w") as fo:
        fo.write("# Forced photometry of the target, aperture %s; fluxes in ADU of the image "
                 "(G = ZP - 2.5 log10 flux)\n" % aperture)
        fo.write("# JD camera field image X Y RA Dec ZP flux_raw eflux_raw flux_corr "
                 "eflux_corr eflux_emp nres\n")
        for i in range(n):
            fo.write("%.6f %s %s %s %.4f %.4f %.6f %.6f %.4f %s %s %s %s %s %d\n"
                     % (jd[i], tab["camera"][i], tab["field"][i],
                        os.path.basename(tab["image"][i]), X[i], Y[i], ra[i], dec[i], zp[i],
                        _fmt(P["raw"][i, 0]), _fmt(P["eraw"][i, 0]), _fmt(P["corr"][i, 0]),
                        _fmt(P["ecorr"][i, 0]), _fmt(P["eemp"][i, 0]), P["nres"][i, 0]))
    with open(os.path.join(outdir, "lightcurve_controls.txt"), "w") as fo:
        fo.write("# image_index JD control dx dy ZP flux_raw eflux_raw flux_corr eflux_corr "
                 "eflux_emp nres\n")
        for i in range(n):
            for j in range(1, len(offsets) + 1):
                fo.write("%d %.6f %d %.2f %.2f %.4f %s %s %s %s %s %d\n"
                         % (i, jd[i], j, offsets[j - 1][0], offsets[j - 1][1], zp[i],
                            _fmt(P["raw"][i, j]), _fmt(P["eraw"][i, j]),
                            _fmt(P["corr"][i, j]), _fmt(P["ecorr"][i, j]),
                            _fmt(P["eemp"][i, j]), P["nres"][i, j]))

    seasons = combine.split_seasons(jd, season_gap)
    labels = [l for l, _ in seasons] + ["all"]
    sidx = dict(seasons)
    sidx["all"] = np.argsort(jd)
    base = np.isfinite(P["corr"][:, 0]) & np.isfinite(P["raw"][:, 0])
    results = {}
    for lab in labels:
        sel = np.zeros(n, bool)
        sel[sidx[lab]] = True
        t, cm, e, ng = combine.lc_season_stats(P["corr"], E3, scale, sel & base, snr_clip)
        results[(LC_METHOD, lab)] = dict(flux=t, ctlmean=cm, ctlrms=e, n=ng)

    # ---- image stacks -----------------------------------------------------------
    raw, eraw = P["raw"][:, 0], P["eraw"][:, 0]
    good = np.isfinite(raw) & np.isfinite(eraw) & (eraw > 0) & np.isfinite(zp)
    nvalid = good.sum()
    good &= ~(np.abs(raw / np.where(eraw > 0, eraw, 1)) > snr_clip)
    say("# image stacks: %d images with valid photometry, %d after the |S/N| <= %g cut"
        % (nvalid, good.sum(), snr_clip))
    x0, y0 = X - xy_offset, Y - xy_offset
    cache = os.path.join(workdir, "cutouts_rescorr.npz")
    key = repr((H, [round(float(v), 6) for v in np.ravel(offsets)], xy_offset,
                track_mask_radius, nmin, [os.path.basename(f) for f in tab["image"]],
                [round(float(v), 4) for v in X], [round(float(v), 4) for v in Y]))
    d = np.load(cache) if os.path.exists(cache) else None
    if d is not None and str(d["key"]) == key:
        cut, msk, ok = d["cut"], d["msk"], d["ok"]
        say("# using cached cutouts %s" % cache)
    else:
        cut, msk, ok = imstack.residual_corrected_cutouts(tab, x0, y0, offsets, H,
                                                          track_mask_radius, nmin)
        np.savez_compressed(cache, cut=cut, msk=msk, ok=ok, key=np.array(key))
    wap = imstack.aperture_weights(S, H, H, r_ap)
    fmask = (msk[:, 0] * wap[None]).sum((1, 2)) / wap.sum()
    nlack = (good & ~ok[:, 0]).sum()
    good &= ok[:, 0] & (fmask <= max_mask_frac)
    say("# image stacks: %d images lack the residual correction at the target; "
        "%d usable" % (nlack, good.sum()))
    pixsig = imstack.pixel_noise(cut, msk, ok, H)

    if resample_fluxcorr is None:
        if "calimage" not in tab:
            raise RuntimeError("resample_fluxcorr=None needs a 'calimage' column")
        vals = []
        gi = np.where(good)[0]
        for i in gi[::max(1, len(gi) // nres_check)][:nres_check]:
            v, ns = imstack.measure_resample_loss(tab["calimage"][i], r_ap, rin, rwid, H)
            vals.append(v)
            say("# resampling flux factor from %d stars in %s: %.3f"
                % (ns, os.path.basename(tab["calimage"][i]), v))
        rcorr = float(np.median(vals))
    else:
        rcorr = float(resample_fluxcorr)
    say("# resampling flux factor: %.3f (image-stack fluxes are divided by it)" % rcorr)

    idx = np.where(good)[0]
    img_seasons = combine.split_seasons(jd[idx], season_gap)
    igroups = [(lab, idx[s]) for lab, s in img_seasons] + [("all", idx)]
    stamps = {}
    stackdir = os.path.join(outdir, "image_stacks")
    os.makedirs(stackdir, exist_ok=True)
    for lab, ii in igroups:
        stk = []
        for j in range(1 + len(offsets)):
            use = ok[ii, j] & np.isfinite(pixsig[ii, j]) & (pixsig[ii, j] > 0)
            jj = ii[use]
            stk.append(imstack.stack(cut[jj, j] * scale[jj, None, None], msk[jj, j],
                                     pixsig[jj, j] * scale[jj], clip=pix_clip))
        f = [imstack.aperphot(m, mm, H, H, r_ap, rin, rwid)[0] / rcorr
             for m, e, nu, mm in stk]
        t, ctl = f[0], np.array(f[1:])
        results[(IMG_METHOD, lab)] = dict(flux=t, ctlmean=ctl.mean(), ctlrms=ctl.std(),
                                          n=len(ii))
        stamps[lab] = (stk[0][0], H, (t - ctl.mean()) / ctl.std(), len(ii))
        hdr = fits.Header()
        hdr["OBJECT"] = "shift-and-stack, season %s" % lab
        hdr["NIMAGES"] = len(ii)
        hdr["JDMIN"] = jd[ii].min()
        hdr["JDMAX"] = jd[ii].max()
        hdr["ZPREF"] = (zpref, "G = ZPREF - 2.5 log10(flux / RESAMPF)")
        hdr["RESAMPF"] = (rcorr, "resampling flux factor")
        hdr["APFLUX"] = (t, "aperture flux at centre, resampling-corrected")
        hdr["CTRX"] = (H + 1, "FITS pixel of the target (1-based)")
        hdr["CTRY"] = (H + 1, "FITS pixel of the target (1-based)")
        hl = fits.HDUList([fits.PrimaryHDU(stk[0][0].astype(np.float32), hdr),
                           fits.ImageHDU(stk[0][1].astype(np.float32), name="ERR"),
                           fits.ImageHDU(stk[0][2].astype(np.int16), name="NUSED")])
        for j in range(1, 1 + len(offsets)):
            hl.append(fits.ImageHDU(stk[j][0].astype(np.float32), name="CTRL%d" % j))
        hl.writeto(os.path.join(stackdir, "stack_%s.fits" % lab), overwrite=True)

    # ---- table --------------------------------------------------------------------
    for (meth, lab), r in results.items():
        s = sidx[lab]
        r["jdmin"], r["jdmax"] = jd[s].min(), jd[s].max()
        r["sig"] = (r["flux"] - r["ctlmean"]) / r["ctlrms"]
        r["G"], r["egp"], r["egm"] = combine.magnitude_with_errors(r["flux"], r["ctlrms"],
                                                                   zpref)
        r["Vpred"] = np.mean(np.interp(jd[s], hjd, hmag))
    with open(os.path.join(outdir, "seasonal_photometry.txt"), "w") as fo:
        fo.write("# aperture %s; G = ZPref - 2.5 log10(flux), ZPref = %.4f, flux in ADU at "
                 "ZPref\n" % (aperture, zpref))
        fo.write("# ctlmean, ctlrms = mean and rms over %d control positions; "
                 "sig = (flux - ctlmean)/ctlrms; eG from ctlrms (+ = faint side)\n"
                 % len(offsets))
        fo.write("# method | season | JDmin JDmax | n | flux ctlmean ctlrms sig | "
                 "G eG(+) eG(-) | Horizons V (mean)\n")
        for meth in (LC_METHOD, IMG_METHOD):
            for lab in labels:
                r = results.get((meth, lab))
                if r is None:
                    continue
                fo.write("%-66s | %-4s | %.2f %.2f | %4d | %6.2f %6.2f %5.2f %5.1f | "
                         "%6.2f +%.2f -%.2f | %.2f\n"
                         % (meth, lab, r["jdmin"], r["jdmax"], r["n"], r["flux"],
                            r["ctlmean"], r["ctlrms"], r["sig"], r["G"], r["egp"], r["egm"],
                            r["Vpred"]))
    say(open(os.path.join(outdir, "seasonal_photometry.txt")).read())

    if make_plot:
        figure.make_figure(os.path.join(outdir, "figure"), seasons, jd, stamps, results,
                           hjd, hra, hdec, hmag, LC_METHOD, IMG_METHOD)
    say("# done in %.0f s; outputs in %s" % (time.time() - t0, outdir))
    log.close()
    return results
