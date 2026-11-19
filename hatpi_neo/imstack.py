"""
Image-based shift-and-stack with a static residual correction.

  1. Images of the same camera and field share the reference image and
     pixel grid, so subtraction residuals fixed on the sky recur at the
     same pixels. For each (camera, field) group a static residual image
     is built: at each pixel, the median and MAD sigma over the group's
     images are computed (excluding masked pixels and pixels within
     track_mask_radius of the target in that image), values within
     4 sigma of the median are kept, and for image i the residual is the
     mean of the kept values of the other images (leave-one-out). Pixels
     with fewer than nmin contributing images are masked.
  2. The residual image is subtracted, and (2H+1)^2 cutouts are taken at
     the target and at each control position, shifted with bilinear
     interpolation so the position lands on the centre of pixel (H, H).
     Masked pixels (EXPCOUNT == 0 in the HATPI stacks, or non-finite) are
     propagated through the shift.
  3. Cutouts are scaled to a common zero point, and combined with an
     inverse-variance weighted mean (weight 1/(pixsig * scale)^2, pixsig =
     robust pixel noise of the cutout outside r = 4 px) with iterative
     per-pixel sigma clipping.
  4. Aperture photometry (fractional-pixel circular aperture, median
     annulus background) is done at the centre of the stacked images; the
     flux is divided by the fraction of a point source's aperture flux
     kept after the bilinear resampling.
"""

import math
import sys
import time
import warnings

import numpy as np
from astropy.io import fits
from scipy import ndimage

from .inputs import image_hdu


def read_image(fname):
    """Difference image and bad-pixel mask (EXPCOUNT == 0 if that extension
    exists, plus non-finite pixels)."""
    with fits.open(fname) as h:
        img = h[image_hdu(h)].data
        bad = (h["EXPCOUNT"].data == 0) if "EXPCOUNT" in h else np.zeros(img.shape, bool)
    bad = bad | ~np.isfinite(img)
    return np.nan_to_num(img).astype(np.float32), bad


def shifted_cutout(img, mask, x0, y0, H):
    """(2H+1)^2 cutout with the 0-based pixel-centre position (x0, y0)
    placed on the centre of pixel (H, H) by a bilinear shift. Returns
    (data, mask), or (None, None) if too close to the edge."""
    ny, nx = img.shape
    ix, iy = int(np.floor(x0 + 0.5)), int(np.floor(y0 + 0.5))
    P = H + 2
    if ix - P < 0 or iy - P < 0 or ix + P + 1 > nx or iy + P + 1 > ny:
        return None, None
    sub = img[iy - P:iy + P + 1, ix - P:ix + P + 1].astype(np.float64)
    msk = mask[iy - P:iy + P + 1, ix - P:ix + P + 1].astype(np.float64)
    sub = np.where(msk > 0, 0.0, sub)
    dx, dy = ix - x0, iy - y0
    s = ndimage.shift(sub, (dy, dx), order=1, mode="constant", cval=0.0)
    m = ndimage.shift(msk, (dy, dx), order=1, mode="constant", cval=1.0)
    s = s[2:-2, 2:-2]
    m = m[2:-2, 2:-2] > 1e-3
    return s.astype(np.float32), m


def residual_corrected_cutouts(tab, x0, y0, offsets, H, track_mask_radius=8.0, nmin=5):
    """Static-residual-corrected cutouts at the target (index 0) and the
    control positions of every image. x0, y0 are 0-based pixel-centre
    coordinates of the target. Returns cut (n, npos, S, S), msk, ok (n, npos)."""
    n = len(x0)
    allpos = [(0.0, 0.0)] + list(offsets)
    npos = len(allpos)
    S = 2 * H + 1
    cut = np.zeros((n, npos, S, S), np.float32)
    msk = np.ones((n, npos, S, S), bool)
    ok = np.zeros((n, npos), bool)
    groups = {}
    for i in range(n):
        groups.setdefault((tab["camera"][i], tab["field"][i]), []).append(i)
    pad = H + 4
    t0 = time.time()
    ndone = 0
    for (cam, field), gi in sorted(groups.items()):
        gx, gy = x0[gi], y0[gi]
        px = np.concatenate([gx + ox for ox, oy in allpos])
        py = np.concatenate([gy + oy for ox, oy in allpos])
        imgs, bads = [], []
        for i in gi:
            img, bad = read_image(tab["image"][i])
            if not imgs:
                ny, nx = img.shape
                bx0 = max(0, int(np.floor(px.min())) - pad)
                bx1 = min(nx, int(np.ceil(px.max())) + pad + 1)
                by0 = max(0, int(np.floor(py.min())) - pad)
                by1 = min(ny, int(np.ceil(py.max())) + pad + 1)
            imgs.append(img[by0:by1, bx0:bx1].copy())
            bads.append(bad[by0:by1, bx0:bx1].copy())
        A = np.array(imgs)
        B = np.array(bads)
        del imgs, bads
        ng = len(gi)
        yy, xx = np.mgrid[by0:by1, bx0:bx1]
        use = ~B
        for k in range(ng):
            use[k] &= np.hypot(xx - gx[k], yy - gy[k]) > track_mask_radius
        V = np.where(use, A, np.nan)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(V, axis=0)
            sig = 1.4826 * np.nanmedian(np.abs(V - med[None]), axis=0)
            clip = use & (np.abs(A - med[None]) < 4 * np.maximum(sig[None], 1e-6))
        del V
        Ssum = np.where(clip, A, 0).sum(0, dtype=np.float64)
        Ncnt = clip.sum(0)
        nres0 = []
        for k in range(ng):
            i = gi[k]
            sk = Ssum - np.where(clip[k], A[k], 0)
            nk = Ncnt - clip[k]
            res = np.where(nk >= nmin, sk / np.maximum(nk, 1), 0.0)
            corr = (A[k] - res).astype(np.float32)
            badk = B[k] | (nk < nmin)
            for j, (ox, oy) in enumerate(allpos):
                xc, yc = gx[k] + ox - bx0, gy[k] + oy - by0
                c, m = shifted_cutout(corr, badk, xc, yc, H)
                if c is not None:
                    cut[i, j], msk[i, j], ok[i, j] = c, m, True
                    if j == 0:
                        nres0.append(nk[int(round(yc)), int(round(xc))])
        ndone += ng
        print("  residual image %s %s: %d images, median N(residual) at target %.0f  "
              "[%d/%d images, %.0f s]" % (cam, field, ng, np.median(nres0) if nres0 else 0,
                                          ndone, n, time.time() - t0), file=sys.stderr)
        del A, B, clip
    return cut, msk, ok


def aperture_weights(S, xc, yc, r, oversamp=10):
    """Fractional-pixel circular aperture weights by sub-sampling."""
    o = (np.arange(oversamp) + 0.5) / oversamp - 0.5
    yy, xx = np.mgrid[0:S, 0:S]
    w = np.zeros((S, S))
    for a in o:
        for b in o:
            w += ((xx + a - xc) ** 2 + (yy + b - yc) ** 2) <= r * r
    return w / oversamp ** 2


def robust_sigma(v):
    v = v[np.isfinite(v)]
    if v.size < 3:
        return np.nan
    return 1.4826 * np.median(np.abs(v - np.median(v)))


def aperphot(img, mask, xc, yc, r, rin, rwid):
    """Aperture flux with a median-annulus background. Returns
    (flux, formal_err, bg, bgsig, fraction of the aperture masked)."""
    S = img.shape[0]
    w = aperture_weights(S, xc, yc, r)
    yy, xx = np.mgrid[0:S, 0:S]
    rr = np.hypot(xx - xc, yy - yc)
    ann = (rr >= rin) & (rr < rin + rwid) & ~mask
    if ann.sum() < 10:
        return np.nan, np.nan, np.nan, np.nan, 1.0
    bg = np.median(img[ann])
    bgsig = robust_sigma(img[ann])
    area = w.sum()
    fmask = (w * mask).sum() / area
    flux = (w * np.where(mask, 0.0, img - bg)).sum()
    err = bgsig * math.sqrt(area * (1.0 + area / ann.sum()))
    return flux, err, bg, bgsig, fmask


def measure_resample_loss(calfile, r_ap, rin, rwid, H, nstar=400):
    """Median ratio of the aperture flux after the bilinear shift-to-centre
    to that in an exact fractional aperture on the unshifted image, for
    unsaturated stars in a calibrated (non-subtracted) image."""
    with fits.open(calfile) as h:
        img = np.nan_to_num(h[image_hdu(h)].data).astype(np.float64)
    ny, nx = img.shape
    med = np.median(img)
    sm = ndimage.uniform_filter(img, 3)
    mx = ndimage.maximum_filter(sm, 9)
    pk = np.argwhere((sm == mx) & (sm > med + 300) & (sm < med + 3000))
    pk = pk[(pk[:, 0] > 200) & (pk[:, 0] < ny - 200) &
            (pk[:, 1] > 200) & (pk[:, 1] < nx - 200)][:nstar]
    bad = np.zeros(img.shape, bool)
    yy, xx = np.mgrid[-3:4, -3:4]
    rat = []
    for (y, x) in pk:
        s = img[y - 3:y + 4, x - 3:x + 4] - np.median(img[y - 15:y + 16, x - 15:x + 16])
        s = np.clip(s, 0, None)
        if s.sum() <= 0:
            continue
        cx = x + (s * xx).sum() / s.sum()
        cy = y + (s * yy).sum() / s.sum()
        ix, iy = int(round(cx)), int(round(cy))
        sub = img[iy - H:iy + H + 1, ix - H:ix + H + 1]
        f0 = aperphot(sub, bad[iy - H:iy + H + 1, ix - H:ix + H + 1],
                      H + cx - ix, H + cy - iy, r_ap, rin, rwid)[0]
        c, m = shifted_cutout(img, bad, cx, cy, H)
        if c is None or not f0 > 0:
            continue
        rat.append(aperphot(c, m, H, H, r_ap, rin, rwid)[0] / f0)
    return np.median(rat), len(rat)


def stack(cuts, msks, sig, clip=4.0, niter=3):
    """Inverse-variance weighted mean with iterative per-pixel sigma
    clipping. cuts: (n, S, S) scaled cutouts; sig: (n,) per-pixel noise.
    Returns (mean, err, nused, mask)."""
    w0 = 1.0 / sig[:, None, None] ** 2 * (~msks)
    use = ~msks
    for it in range(niter + 1):
        w = w0 * use
        sw = w.sum(0)
        mean = np.where(sw > 0, (w * cuts).sum(0) / np.where(sw > 0, sw, 1), 0)
        if it == niter:
            break
        res = (cuts - mean[None]) / sig[:, None, None]
        new = (~msks) & (np.abs(res) < clip)
        if np.array_equal(new, use):
            break
        use = new
    err = np.where(sw > 0, 1.0 / np.sqrt(np.where(sw > 0, sw, 1)), np.inf)
    return mean, err, use.sum(0), sw == 0


def pixel_noise(cut, msk, ok, H):
    """Robust per-pixel noise of every cutout, excluding r <= 4 px."""
    S = 2 * H + 1
    yy, xx = np.mgrid[0:S, 0:S]
    outer = np.hypot(xx - H, yy - H) > 4
    pixsig = np.full(ok.shape, np.nan)
    for i, j in zip(*np.where(ok)):
        pixsig[i, j] = robust_sigma(cut[i, j][outer & ~msk[i, j]])
    return pixsig
