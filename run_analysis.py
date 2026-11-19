#!/usr/bin/env python3
"""Command-line wrapper for hatpi_neo.pipeline.run_analysis().

Example (the settings used for 2026 ED138 are the defaults):
    python3 run_analysis.py --horizons examples/2026ED138/horizons_results_2026ED138.txt \\
        --images image_list.txt --outdir results_2026ED138 --fiphot /path/to/fiphot
"""

import argparse

from hatpi_neo.pipeline import run_analysis


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--horizons", required=True, help="JPL Horizons observer table")
    p.add_argument("--images", required=True, help="image list (see README)")
    p.add_argument("--outdir", required=True, help="output directory")
    p.add_argument("--fiphot", default="fiphot", help="fiphot executable (0.9zc or later)")
    p.add_argument("--aperture", default="1.45:8:8",
                   help="aperture radius:annulus inner radius:annulus width, pixels "
                        "(default %(default)s)")
    p.add_argument("--ncontrol", type=int, default=10,
                   help="control positions per ring (default %(default)s)")
    p.add_argument("--control-radii", default="50,90",
                   help="control ring radii, pixels (default %(default)s)")
    p.add_argument("--track-mask-radius", type=float, default=8.0)
    p.add_argument("--nmin", type=int, default=5)
    p.add_argument("--snr-clip", type=float, default=5.0)
    p.add_argument("--pix-clip", type=float, default=4.0)
    p.add_argument("--half-size", type=int, default=40)
    p.add_argument("--xy-offset", type=float, default=0.5)
    p.add_argument("--resample-fluxcorr", default="0.901",
                   help="resampling flux factor, or 'measure' to measure it on the "
                        "calimage files (default %(default)s)")
    p.add_argument("--season-gap", type=float, default=60.0)
    p.add_argument("--nthreads", type=int, default=8)
    p.add_argument("--no-plot", action="store_true")
    a = p.parse_args()
    run_analysis(a.horizons, a.images, a.outdir, fiphot=a.fiphot, aperture=a.aperture,
                 ncontrol=a.ncontrol,
                 control_radii=[float(v) for v in a.control_radii.split(",")],
                 track_mask_radius=a.track_mask_radius, nmin=a.nmin, snr_clip=a.snr_clip,
                 pix_clip=a.pix_clip, half_size=a.half_size, xy_offset=a.xy_offset,
                 resample_fluxcorr=None if a.resample_fluxcorr == "measure"
                 else float(a.resample_fluxcorr),
                 season_gap=a.season_gap, nthreads=a.nthreads, make_plot=not a.no_plot)


if __name__ == "__main__":
    main()
