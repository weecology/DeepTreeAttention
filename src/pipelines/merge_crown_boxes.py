"""Merge per-plot ``*_boxes.shp`` outputs from SLURM workers into a single crowns GeoDataFrame file."""
from __future__ import annotations

import argparse
import glob
import os

import geopandas as gpd
import pandas as pd


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Concatenate plot-level crown box shapefiles into crowns.shp for TreeData (replace=False path)."
    )
    parser.add_argument("--boxes-dir", required=True, help="Directory containing <plot>_boxes.shp merged stem boxes.")
    parser.add_argument("--out", required=True, help="Output path, e.g. data/interim/run/crowns.shp")
    args = parser.parse_args(argv)

    paths = sorted(glob.glob(os.path.join(args.boxes_dir, "*_boxes.shp")))
    if not paths:
        raise FileNotFoundError("No *_boxes.shp files found in --boxes-dir.")

    parts = [gpd.read_file(p) for p in paths]
    merged = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=parts[0].crs)
    merged = merged.drop_duplicates(subset=["plotID", "box_id"], keep="first")
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    merged.to_file(args.out)


if __name__ == "__main__":
    main()
