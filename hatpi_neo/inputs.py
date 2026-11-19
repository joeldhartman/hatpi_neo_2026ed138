"""
Inputs: the JPL Horizons observer table and the list of images.

Image list
----------
A whitespace-separated text table. Lines starting with '#' are comments;
the first non-comment line gives the column names. One row per
(stacked) difference image. Required columns:

  image    difference (subtracted) image, FITS or fpack-compressed FITS
  kernel   convolution-kernel file of that image (fitsh/ficonv format, as
           used by fiphot --input-kernel)
  photref  photometric reference image the difference image was made
           against (fiphot measures reference fluxes on it)
  wcsref   FITS file whose header holds the astrometric (WCS) solution of
           the pixel grid of the difference image
  zp       photometric zero point of the image for the aperture used:
           G = zp - 2.5 log10(flux), flux in ADU in the difference image
  camera   camera / unit identifier (HATPI IHU number)
  field    pointing / field identifier

Optional columns:

  jd       mid-exposure JD (UTC) of the image; if absent, the JD header
           keyword of the image is used (2400000 is added if JD < 2400000)
  calimage the calibrated (non-subtracted) counterpart of the image; only
           used to measure the flux lost by the sub-pixel resampling of
           the cutouts (otherwise a fixed factor is used)

Relative paths are taken relative to the directory of the image list.
Images that share camera, field and photref are on the same pixel grid
and difference-imaged against the same reference.
"""

import os
import re
import warnings

import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS, FITSFixedWarning

REQUIRED = ["image", "kernel", "photref", "wcsref", "zp", "camera", "field"]
PATHCOLS = ["image", "kernel", "photref", "wcsref", "calimage"]


# --------------------------------------------------------------------------
# Horizons
# --------------------------------------------------------------------------

def read_horizons(fname):
    """Read a JPL Horizons observer table (text output, calendar UT dates,
    RA/Dec in HMS/DMS, APmag column requested).

    Returns jd, ra, dec, apmag arrays. RA is unwrapped (continuous across
    0h), in degrees; it is reduced to [0, 360) by target_positions().
    The first RA/Dec pair on each line (astrometric ICRF) is used."""
    lines = open(fname).read().splitlines()
    hdr = None
    isoe = None
    for i, l in enumerate(lines):
        if l.startswith("$$SOE"):
            isoe = i
            break
        if "APmag" in l and "Date" in l:
            hdr = l
    if isoe is None:
        raise RuntimeError("no $$SOE marker in Horizons file %s" % fname)
    if hdr is None:
        raise RuntimeError("no APmag column in Horizons file %s" % fname)
    col_end = hdr.index("APmag") + len("APmag")
    col_start = col_end - 8
    pat = re.compile(r"(\d{2}) (\d{2}) (\d{2}\.\d+) ([-+]\d{2}) (\d{2}) (\d{2}\.\d+)")
    jd, ra, dec, mag = [], [], [], []
    for l in lines[isoe + 1:]:
        if l.startswith("$$EOE"):
            break
        t = l.split()
        m = pat.search(l)
        if not m:
            continue
        h, mi, s, d, dm, ds = m.groups()
        sgn = -1 if d.startswith("-") else 1
        jd.append(Time.strptime(t[0] + " " + t[1], "%Y-%b-%d %H:%M", scale="utc").jd)
        ra.append(15 * (int(h) + int(mi) / 60 + float(s) / 3600))
        dec.append(sgn * (abs(int(d)) + int(dm) / 60 + float(ds) / 3600))
        mm = l[col_start:col_end + 2].split()
        try:
            mag.append(float(mm[-1]) if mm else np.nan)
        except ValueError:
            mag.append(np.nan)
    ra = np.degrees(np.unwrap(np.radians(ra)))
    return np.array(jd), ra, np.array(dec), np.array(mag)


# --------------------------------------------------------------------------
# Image list
# --------------------------------------------------------------------------

def image_hdu(h):
    """Index of the first HDU holding a 2-D image (handles fpack files)."""
    for k, hdu in enumerate(h):
        if hdu.data is not None and getattr(hdu.data, "ndim", 0) == 2:
            return k
    raise RuntimeError("no 2-D image in %s" % h.filename())


def read_image_list(fname):
    """Read the image list; returns a dict of numpy arrays (one per column)."""
    base = os.path.dirname(os.path.abspath(fname))
    names, rows = None, []
    for l in open(fname):
        if not l.strip() or l.lstrip().startswith("#"):
            continue
        t = l.split()
        if names is None:
            names = t
            continue
        if len(t) != len(names):
            raise RuntimeError("%s: row has %d columns, header has %d:\n%s"
                               % (fname, len(t), len(names), l))
        rows.append(t)
    missing = [c for c in REQUIRED if c not in names]
    if missing:
        raise RuntimeError("%s: missing required column(s) %s" % (fname, ", ".join(missing)))
    tab = {c: [r[k] for r in rows] for k, c in enumerate(names)}
    for c in PATHCOLS:
        if c in tab:
            tab[c] = [p if os.path.isabs(p) else os.path.join(base, p) for p in tab[c]]
    out = {c: np.array(v, dtype=object) for c, v in tab.items()}
    out["zp"] = np.array(tab["zp"], float)
    out["camera"] = np.array(tab["camera"], dtype=str)
    out["field"] = np.array(tab["field"], dtype=str)
    if "jd" in tab:
        out["jd"] = np.array(tab["jd"], float)
    else:
        jd = []
        for f in out["image"]:
            with fits.open(f) as h:
                v = float(h[image_hdu(h)].header["JD"])
            jd.append(v + 2400000.0 if v < 2400000.0 else v)
        out["jd"] = np.array(jd)
    return out


# --------------------------------------------------------------------------
# Positions
# --------------------------------------------------------------------------

def target_positions(tab, hjd, hra, hdec):
    """Predicted RA, Dec (linear interpolation of the Horizons table to the
    image JD) and pixel position X, Y of the target in every image.

    X, Y are computed with astropy WCS.all_world2pix(..., origin=1) from the
    wcsref header and are passed to fiphot unchanged; the image-stacking
    code converts them to 0-based pixel-centre coordinates by subtracting
    xy_offset (0.5, the fitsh convention; see imstack.py)."""
    jd = tab["jd"]
    ra = np.mod(np.interp(jd, hjd, hra), 360.0)
    dec = np.interp(jd, hjd, hdec)
    x = np.full(len(jd), np.nan)
    y = np.full(len(jd), np.nan)
    wcs_cache = {}
    for i, f in enumerate(tab["wcsref"]):
        if f not in wcs_cache:
            with fits.open(f) as h, warnings.catch_warnings():
                warnings.simplefilter("ignore", FITSFixedWarning)
                wcs_cache[f] = WCS(h[image_hdu(h)].header)
        xp, yp = wcs_cache[f].all_world2pix(ra[i], dec[i], 1)
        x[i], y[i] = float(xp), float(yp)
    return ra, dec, x, y


def control_offsets(nctrl, radii):
    """nctrl control positions on each ring of the given radii (pixels);
    successive rings are rotated by half a step. Returns [(dx, dy), ...]."""
    offs = []
    for k, radius in enumerate(radii):
        ang = 2 * np.pi * (np.arange(nctrl) + 0.5 * k) / nctrl
        offs += [(radius * np.cos(a), radius * np.sin(a)) for a in ang]
    return offs
