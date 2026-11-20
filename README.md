# hatpi_neo: HATPI shift-and-stack photometry of 2026 ED138

This is the analysis code for the HATPI recovery of the trans-Neptunian object
2026 ED138, described in the Research Note of the AAS *"HATPI Detection of
2026 ED138"* (Hartman et al. 2026). Running it on the images listed in
`examples/2026ED138/` produces the per-season photometry quoted in the Note
and its figure. It takes two inputs:

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

## Contents

| file | format | contents |
|---|---|---|
| `README.md` | Markdown | this description |
| `LICENSE` | text | MIT license |
| `CITATION.cff`, `.zenodo.json` | YAML, JSON | citation and archive metadata |
| `requirements.txt` | text | Python package requirements |
| `run_analysis.py` | Python 3 | command-line driver |
| `hatpi_neo/inputs.py` | Python 3 | reading the Horizons file and image list; target pixel positions |
| `hatpi_neo/lcphot.py` | Python 3 | Light Curve-based method: fiphot forced photometry, residual correction |
| `hatpi_neo/imstack.py` | Python 3 | Image-based method: residual images, shifted cutouts, stacking, aperture photometry |
| `hatpi_neo/combine.py` | Python 3 | seasons, weighted means, control-position statistics, magnitudes |
| `hatpi_neo/figure.py` | Python 3 | the figure of the Note |
| `hatpi_neo/pipeline.py` | Python 3 | `run_analysis()`, which runs all steps and writes the outputs |
| `examples/2026ED138/horizons_results_2026ED138.txt` | text | JPL Horizons ephemeris used in the Note (see Data sources) |
| `examples/2026ED138/stack_metadata.txt` | text table | the 1498 HATPI images used in the Note (columns below) |

The HATPI images themselves are not part of this repository.

## Requirements

- Python ≥ 3.9 with numpy, scipy, astropy and matplotlib (`requirements.txt`).
  The note used numpy 1.21, scipy 1.7, astropy 5.0 and matplotlib 3.5.
- **fiphot** from the fitsh / HATpipe package, version 0.9zc or later. It must
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

### `examples/2026ED138/stack_metadata.txt`

Whitespace-separated, one row per image used in the Note (1498 rows). Each image
is a 3060 s mean-combined stack of 30 s or 45 s HATPI difference images.

| column | units | meaning |
|---|---|---|
| `jd` | days | mid-exposure Julian Date (UTC) |
| `camera` | — | HATPI camera (IHU) number |
| `field` | — | HATPI field name |
| `reference` | YYYYMMDD | date of the photometric/astrometric reference the image was subtracted against |
| `zp` | mag | zero point: G = zp − 2.5 log10(flux / ADU) |
| `image`, `kernel`, `wcsref` | — | file names (no paths) of the difference image, its kernel and the WCS reference |

To run the code, turn it into an image list (see above) with full paths to
your copies of these files.

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

Output columns and units:

- **`lightcurve.txt`**, one row per image:
  - `JD` (days, UTC); `camera`, `field`, `image`.
  - `X`, `Y`: target position (pixels, fitsh/1-based convention).
  - `RA`, `Dec`: target position (degrees, ICRF).
  - `ZP` (mag).
  - `flux_raw`, `eflux_raw`, `flux_corr`, `eflux_corr`, `eflux_emp`: ADU of that
    image. `eflux_emp` is the scatter of the residual estimate.
  - `nres`: number of images in the residual estimate.
- **`lightcurve_controls.txt`**: `image_index` (0-based row of
  `lightcurve.txt`), `JD`, `control` (1–20), `dx`, `dy` (pixels from the
  target), then the same columns as `lightcurve.txt`.
- **`seasonal_photometry.txt`**: method; season; JD range (days); number of
  images; flux, control mean and control rms (ADU at ZPref); significance (σ);
  G and its errors (mag); mean Horizons V (mag).
- **`stack_<season>.fits`**: pixel values in ADU at ZPref, 81×81 pixels
  (≈20″ per pixel), with the target at the centre of pixel (41, 41) (1-based).
  `ERR` is the formal error in the same units; `NUSED` is the number of images
  per pixel (integer).

Runtime for 1498 images on 8 cores is about 30 minutes.

## Data sources

- **HATPI**: images from the HATPI facility at Las Campanas Observatory
  (Bakos et al. 2026, submitted to PASP), reduced with the HATPI pipeline.
- **Ephemeris**: JPL Horizons (Giorgini et al. 1996), orbit solution JPL#2
  (2026 Oct 29), DE441, observer site Las Campanas; retrieved 2026 Nov 9.
- **Zero points**: calibrated against Gaia DR2 G (Gaia Collaboration et al.
  2016, 2018).

## Citation

If you use this code, please cite this archive (Zenodo DOI), the Research Note
"HATPI Detection of 2026 ED138" (Hartman et al. 2026), and the HATPI instrument
paper (Bakos et al. 2026, submitted to PASP).

## License

MIT (see `LICENSE`).
