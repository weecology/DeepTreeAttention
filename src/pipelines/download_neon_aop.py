"""CLI: download NEON AOP tiles overlapping field stem points using neonutilities."""
from __future__ import annotations

import argparse
import os

import geopandas as gpd

from src import utils
from src.neon_download import download_products_for_points


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Download NEON AOP products by tile for coordinates in a point file."
    )
    parser.add_argument("--config", default="config.yml", help="YAML config (uses neon_download section).")
    parser.add_argument(
        "--points",
        help="Override points path (GeoPackage/Shapefile). Defaults to config data_layout.canopy_points_shp or interim path.",
    )
    args = parser.parse_args(argv)

    config = utils.read_config(args.config)
    nd = config.get("neon_download") or {}
    layout = config.get("data_layout") or {}

    points_path = args.points or layout.get("canopy_points_shp")
    if not points_path:
        raise ValueError(
            "No --points given and data_layout.canopy_points_shp missing in config. "
            "Run filtering once to write canopy_points.shp, or pass --points explicitly."
        )

    site = nd.get("site") or "OSBS"
    years = nd.get("years") or [2021]
    buffer = int(nd.get("buffer_m", 50))
    check_size = bool(nd.get("check_size", False))
    products = nd.get("products") or {
        "rgb": "DP3.30010.001",
        "hsi": "DP3.30006.001",
        "chm": "DP3.30015.001",
    }
    save_root = nd.get("save_root") or layout.get("sensor_download_root") or "data/external/neon_aop"

    token_env = nd.get("token_env", "NEON_API_TOKEN")
    token = os.environ.get(token_env)

    gdf = gpd.read_file(points_path)
    download_products_for_points(
        gdf=gdf,
        site=site,
        years=years,
        save_root=save_root,
        products=products,
        buffer=buffer,
        token=token,
        check_size=check_size,
    )


if __name__ == "__main__":
    main()
