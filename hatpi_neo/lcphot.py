"""
Forced aperture photometry of the target on every difference image, with a
correction for static (sky-fixed) subtraction residuals.

Photometry is done with fiphot (fitsh / HATpipe):
for every group of images sharing camera, field and photometric reference,
  1. fiphot measures the reference fluxes on the photometric reference
     image at all positions needed by the group (the target position in
     every image of the group, plus the control positions), and the
     per-aperture flag columns of that raw-photometry file are set to 0;
  2. fiphot --input-subtracted measures the difference flux at those
     positions on every image of the group, using the image's convolution
     kernel and the options --fluxerr-bg-only --bgsigma-mad-correct (the
     error is the background-noise term only).
This gives flux[k, p], eflux[k, p] for image k at position p.

Static residual correction. For image i and position p, the residual R is
estimated from the other images k != i of the group in which the target
was more than track_mask_radius pixels from p: their fluxes at p, on a
common scale g = flux * 10^(-0.4 zp), are clipped at clip x (MAD sigma)
around the median, and R is their mean, converted back to image i's scale.
  flux_corr  = flux[i, p] - R
  eflux_corr = sqrt(eflux[i, p]^2 + s^2 / N)
  eflux_emp  = s      (MAD sigma of those fluxes in image i's scale: an
                       empirical per-position noise estimate)
  nres       = N      (positions with N < nmin are left NaN)
"""

import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from astropy.io import fits

from .inputs import image_hdu


def check_fiphot(fiphot):
    exe = shutil.which(fiphot) or (fiphot if os.access(fiphot, os.X_OK) else None)
    if exe is None:
        raise RuntimeError("cannot find the fiphot executable '%s'" % fiphot)
    r = subprocess.run([exe, "--long-help"], capture_output=True, text=True)
    if "--fluxerr-bg-only" not in r.stdout + r.stderr:
        raise RuntimeError("%s does not support --fluxerr-bg-only; fiphot 0.9zc or later "
                           "is needed" % exe)
    return exe


def _run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def _fnum(s):
    try:
        return float(s)
    except ValueError:
        return np.nan


def uncompressed(fname, tmpdir):
    """Return (path to an uncompressed copy, is_temporary) for fpack files
    (fiphot reads plain FITS). funpack is used if available."""
    if not fname.endswith(".fz"):
        return fname, False
    fd, tmp = tempfile.mkstemp(suffix=".fits", dir=tmpdir)
    os.close(fd)
    os.remove(tmp)
    if shutil.which("funpack"):
        r = _run(["funpack", "-O", tmp, fname])
        if r.returncode == 0:
            return tmp, True
    with fits.open(fname) as h:
        k = image_hdu(h)
        fits.PrimaryHDU(h[k].data, h[k].header).writeto(tmp)
    return tmp, True


def make_group_rawphot(photref, xy, aperture, fiphot, outfile):
    """fiphot on the photometric reference at the given positions; the flag
    column of each aperture (1-based column 12 + 6n) is then set to 0."""
    lst = "".join("%d %.4f %.4f\n" % (k + 1, x, y) for k, (x, y) in enumerate(xy))
    tmp = outfile + ".tmp"
    r = _run([fiphot, photref, "-il", "-", "--col-xy", "2,3", "--col-id", "1",
              "-a", aperture, "-op", tmp], input=lst)
    if r.returncode != 0:
        raise RuntimeError("fiphot on %s failed: %s" % (photref, r.stderr[-500:]))
    nap = len(aperture.split(","))
    with open(tmp) as fi, open(outfile, "w") as fo:
        for l in fi:
            if l.startswith("#") or not l.strip():
                fo.write(l)
                continue
            t = l.split()
            for n in range(nap):
                c = 11 + 6 * n
                if c < len(t):
                    t[c] = "0"
            fo.write(" ".join(t) + "\n")
    os.remove(tmp)


def sub_phot(image, kernel, rawphot, fiphot, subopts, npos, tmpdir):
    """fiphot on one difference image; returns (flux, eflux) of aperture 0
    at the npos positions of the raw-photometry file."""
    fl = np.full(npos, np.nan)
    er = np.full(npos, np.nan)
    use, istmp = uncompressed(image, tmpdir)
    cmd = [fiphot] + subopts.split() + ["--input-subtracted", use, "-ik", kernel,
                                       "--input-raw-photometry", rawphot,
                                       "-o", "-", "-F", "IXY,dfuv"]
    r = _run(cmd)
    if istmp and os.path.exists(use):
        os.remove(use)
    if r.returncode != 0:
        print("fiphot failed on %s: %s" % (image, r.stderr[-300:]), file=sys.stderr)
        return fl, er
    for l in r.stdout.splitlines():
        if l.startswith("#") or not l.strip():
            continue
        t = l.split()
        k = int(t[0]) - 1
        fl[k] = _fnum(t[3])
        er[k] = _fnum(t[4])
    return fl, er


def residual_corrected_photometry(tab, x, y, offsets, workdir, fiphot,
                                  aperture="1.45:8:8",
                                  subopts="--fluxerr-bg-only --bgsigma-mad-correct",
                                  track_mask_radius=8.0, nmin=5, clip=4.0, nthreads=8):
    """Forced photometry at the target (position 0) and the control
    positions (1..len(offsets)) of every image, with the static residual
    correction. Returns a dict of (nimage, npos) arrays:
    raw, eraw, corr, ecorr, eemp, nres.

    Flux matrices are cached per group in workdir and reused if present."""
    os.makedirs(workdir, exist_ok=True)
    tmpdir = tempfile.mkdtemp(prefix="tmp_", dir=workdir)
    n = len(x)
    allpos = [(0.0, 0.0)] + list(offsets)
    npp = len(allpos)
    zp = tab["zp"]
    out = {k: np.full((n, npp), np.nan) for k in ("raw", "eraw", "corr", "ecorr", "eemp")}
    out["nres"] = np.zeros((n, npp), int)

    groups = {}
    for i in range(n):
        groups.setdefault((tab["camera"][i], tab["field"][i], tab["photref"][i]), []).append(i)
    print("# forced photometry: %d groups (camera, field, reference)" % len(groups),
          file=sys.stderr)
    t0 = time.time()
    ndone = 0
    for gnum, ((cam, field, photref), gi) in enumerate(sorted(groups.items())):
        ng = len(gi)
        px = np.concatenate([x[gi] + ox for ox, oy in allpos])
        py = np.concatenate([y[gi] + oy for ox, oy in allpos])
        tag = "%s_%s_ref%03d" % (cam, field, gnum)
        matfile = os.path.join(workdir, "fluxmatrix_%s.npz" % tag)
        key = np.array([str(len(gi)), photref] + [str(tab["image"][i]) for i in gi]
                       + ["%.4f %.4f" % v for v in zip(px, py)])
        FL = None
        if os.path.exists(matfile):
            d = np.load(matfile)
            if d["key"].shape == key.shape and np.all(d["key"] == key):
                FL, ER = d["fl"], d["er"]
        if FL is None:
            rawphot = os.path.join(workdir, "rawphot_%s.txt" % tag)
            make_group_rawphot(photref, list(zip(px, py)), aperture, fiphot, rawphot)

            def job(i):
                return sub_phot(tab["image"][i], tab["kernel"][i], rawphot, fiphot,
                                subopts, len(px), tmpdir)
            with ThreadPoolExecutor(nthreads) as ex:
                res = list(ex.map(job, gi))
            FL = np.array([r[0] for r in res])      # (ng, npos)
            ER = np.array([r[1] for r in res])
            np.savez(matfile + ".tmp.npz", fl=FL, er=ER, key=key)
            os.replace(matfile + ".tmp.npz", matfile)
        sc = 10 ** (-0.4 * zp[gi])                  # to a common scale
        G = FL * sc[:, None]
        tx, ty = x[gi], y[gi]
        for kk, i in enumerate(gi):
            for j in range(npp):
                pidx = j * ng + kk
                far = np.hypot(tx - px[pidx], ty - py[pidx]) > track_mask_radius
                far[kk] = False
                out["raw"][i, j] = FL[kk, pidx]
                out["eraw"][i, j] = ER[kk, pidx]
                v = G[far, pidx]
                v = v[np.isfinite(v)]
                if v.size >= 3:
                    med = np.median(v)
                    sig = 1.4826 * np.median(np.abs(v - med))
                    if sig > 0:
                        v = v[np.abs(v - med) < clip * sig]
                out["nres"][i, j] = v.size
                if v.size < nmin or not np.isfinite(sc[kk]):
                    continue
                R = v.mean() / sc[kk]
                s = (1.4826 * np.median(np.abs(v - np.median(v)))) / sc[kk]
                out["corr"][i, j] = FL[kk, pidx] - R
                out["ecorr"][i, j] = math.sqrt(ER[kk, pidx] ** 2 + s ** 2 / v.size)
                out["eemp"][i, j] = s
        ndone += ng
        print("  group %s %s (%d images): median N(residual) at target %.0f  "
              "[%d/%d images, %.0f s]" % (cam, field, ng, np.median(out["nres"][gi, 0]),
                                          ndone, n, time.time() - t0), file=sys.stderr)
    shutil.rmtree(tmpdir, ignore_errors=True)
    return out
