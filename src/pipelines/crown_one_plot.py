"""Run ``generate.run`` for a single plot (one SLURM array task)."""
from __future__ import annotations

import argparse
import glob
import os

import geopandas as gpd

from src import generate


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="DeepForest crown boxes for one NEON plotID.")
    parser.add_argument("--canopy-points", required=True, help="Path to canopy_points.shp (or .gpkg).")
    parser.add_argument("--plot", required=True, help="plotID value to process.")
    parser.add_argument("--rgb-glob", required=True, help="Recursive glob for RGB tiles (same as config rgb_sensor_pool).")
    parser.add_argument("--savedir", required=True, help="Directory for merged stem boxes (one shapefile per plot).")
    parser.add_argument("--raw-box-savedir", required=True, help="Directory for all detector boxes.")
    args = parser.parse_args(argv)

    df = gpd.read_file(args.canopy_points)
    rgb_pool = glob.glob(args.rgb_glob, recursive=True)
    if not rgb_pool:
        raise FileNotFoundError("No RGB rasters matched --rgb-glob; check paths and config.")

    os.makedirs(args.savedir, exist_ok=True)
    os.makedirs(args.raw_box_savedir, exist_ok=True)

    generate.run(
        plot=args.plot,
        df=df,
        rgb_pool=rgb_pool,
        savedir=args.savedir,
        raw_box_savedir=args.raw_box_savedir,
    )


if __name__ == "__main__":
    main()
