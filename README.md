# hatpi_neo: HATPI shift-and-stack photometry of 2026 ED138

This is the analysis code for the HATPI recovery of the trans-Neptunian object
2026 ED138, as described in the accompanying RNAAS note. It takes two inputs:

- a JPL Horizons ephemeris;
- a list of HATPI difference images.

It then measures the object's brightness in two ways:

1. **Light Curve-based stacking**: forced aperture photometry at the predicted
   position in every image, corrected for static subtraction residuals, then
   combined per observing season as an inverse-variance weighted mean.
2. **Image-based stacking**: residual-corrected cutouts are shifted onto the
   predicted position, then stacked per season and measured with the same
   aperture.

For both methods, the significance and the magnitude uncertainties come from
20 control positions that move with the target.

## Requirements

- Python ≥ 3.9 with numpy, scipy, astropy and matplotlib (`requirements.txt`).
  The note used numpy 1.21, scipy 1.7, astropy 5.0 and matplotlib 3.5.
- **fiphot** from the fitsh / HATpipe package. It must
  support `--fluxerr-bg-only` and `--bgsigma-mad-correct`; the code checks this
  at start-up.
- `funpack` (optional): used to decompress fpack (`.fz`) images for fiphot. If
  it is missing, astropy is used instead.

## Usage

```bash
python3 run_analysis.py --horizons examples/2026ED138/horizons_results_2026ED138.txt \
    --images image_list.txt --outdir results --fiphot /path/to/fiphot
```

From Python:

```python
from hatpi_neo.pipeline import run_analysis
results = run_analysis("horizons.txt", "image_list.txt", "results", fiphot="/path/to/fiphot")
```

The default settings are the ones used in the note. `python3 run_analysis.py -h`
lists the options.

## Inputs

### Horizons file

A JPL Horizons *observer table* in text form, with:

- calendar UT dates (`YYYY-Mon-DD HH:MM`);
- astrometric RA/Dec in HMS/DMS;
- the `APmag` column.

For 2026 ED138 we used Las Campanas Observatory as the observing site and a
1-day step (`examples/2026ED138/horizons_results_2026ED138.txt`). Positions are
linearly interpolated to each image's JD.

### Image list

A whitespace-separated table. The first non-comment line gives the column
names, and there is one row per difference image. Relative paths are taken
relative to the list file.

| column | meaning |
|---|---|
| `image` | difference image (FITS or fpack `.fz`). Pixels with `EXPCOUNT`=0 (if that extension exists) or non-finite values are treated as masked. |
| `kernel` | the image's convolution-kernel file (fiphot `-ik`). |
| `photref` | photometric reference image the difference image was made against. |
| `wcsref` | FITS file whose header holds the WCS of the image's pixel grid. |
| `zp` | zero point for the aperture: G = zp − 2.5 log10(flux in ADU). |
| `camera`, `field` | camera (HATPI IHU) and field identifiers. |
| `jd` *(optional)* | mid-exposure JD (UTC). If absent, the image's `JD` header keyword is used. |
| `calimage` *(optional)* | calibrated (non-subtracted) counterpart. Only needed with `--resample-fluxcorr measure`. |

Images with the same `camera`, `field` and `photref` share a reference image
and pixel grid.

- The light-curve residual correction is computed within such groups.
- The image-based residual correction is computed per (`camera`, `field`).

### Zero points

The zero points are an input; this code does not compute them. For the note
they come from the HATPI reduction:

- the photometric reference was calibrated against Gaia DR2 G with a 5th-order
  polynomial in x, y plus a BP−RP colour term;
- that solution was evaluated at the target position, assuming BP−RP = 1.0 for
  the target.

The colour coefficient is about 0.11 mag/mag.

`examples/2026ED138/stack_metadata.txt` lists, for each of the 1498 images used
in the note:

- the JD, camera, field and reference date;
- the zero point;
- the image, kernel and WCS file names.

## Method details

- **Positions.** X, Y = `WCS.all_world2pix(RA, Dec, 1)` from the `wcsref`
  header. These are passed to fiphot unchanged. The image code uses X − 0.5 and
  Y − 0.5 as 0-based pixel-centre coordinates (`--xy-offset`, fitsh convention).
- **Control positions.** 10 positions on each of two rings, 50 and 90 px from
  the target. The outer ring is rotated by half a step.
- **Aperture.** Radius 1.45 px, with the background annulus from 8 to 16 px
  (`1.45:8:8`).
- **Light-curve photometry** (`hatpi_neo/lcphot.py`):
  - fiphot measures the reference fluxes on `photref`. It is then run with
    `--input-subtracted` and the kernel, using `--fluxerr-bg-only
    --bgsigma-mad-correct`.
  - The static residual for an image at a position is the 4σ-clipped (MAD)
    mean flux at that position in the other images of its group, using only
    images whose target was more than 8 px away. At least 5 images are
    required.
  - The per-image error is max(√(σ² + s²/N), s), where s is the scatter of
    those fluxes.
- **Image stacks** (`hatpi_neo/imstack.py`):
  - The static residual image is the leave-one-out, 4σ-clipped per-pixel mean
    of the other images of the same camera and field. Pixels within 8 px of
    the target are masked, and at least 5 images are required.
  - The image is shifted with a bilinear sub-pixel shift and cut to 81×81 px.
  - Images are combined with weights 1/(pixel σ × scale)² and 4σ per-pixel
    clipping.
  - Fluxes are divided by 0.901, the fraction of a point source's aperture
    flux kept after the bilinear resampling. This was measured on stars in
    HATPI calibrated stacks (`--resample-fluxcorr measure` repeats the
    measurement).
- **Selection.** Images with |S/N| ≥ 5 at a position are excluded at that
  position, to reject subtraction residuals.
  - The image stacks use the uncorrected photometry for this cut, and also
    exclude images with masked pixels in the aperture or without a residual
    correction at the target.
- **Combination** (`hatpi_neo/combine.py`):
  - Seasons are separated by gaps of more than 60 days.
  - Fluxes are put on a common zero point, ZPref = the median zp.
  - Significance = (target − mean of controls) / rms of controls. The
    magnitude errors also come from the rms of the controls.

## Outputs (in `--outdir`)

| file | contents |
|---|---|
| `lightcurve.txt` | per image: JD, position, ZP, raw and residual-corrected flux and errors |
| `lightcurve_controls.txt` | the same at the 20 control positions |
| `image_stacks/stack_<season>.fits` | stacked image per season and for all data (target in the primary HDU; ERR, NUSED, CTRL1..20 extensions) |
| `seasonal_photometry.txt` | per method and season: N, flux, control mean and rms, significance, G with errors, Horizons V |
| `figure.png`, `figure.pdf` | the figure of the note |
| `run_log.txt` | log |
| `work/` | cached photometry and cutouts; delete it to recompute |

Runtime for 1498 images on 8 cores is about 30 minutes.

## Citation

If you use this code, please cite the RNAAS note (Hartman et al., 2026) and
the HATPI instrument paper (Bakos et al., submitted to PASP).

## Generative AI Usage

The code in this repository was developed using Claude Code together with the Claude Opus 5.5 model.
