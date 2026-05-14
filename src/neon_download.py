"""NEON AOP downloads via ``neonutilities`` (Python port of R neonUtilities)."""
from __future__ import annotations


import os
from typing import Iterable, List, Sequence

import geopandas as gpd

try:
    import neonutilities as nu
except ImportError as exc:  # pragma: no cover - optional until uv sync
    nu = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _require_neonutilities():
    if nu is None:
        raise ImportError(
            "The neonutilities package is required for downloads. "
            "Install project dependencies (e.g. uv sync) and retry."
        ) from _IMPORT_ERROR


def easting_northing_from_geodataframe(gdf: gpd.GeoDataFrame) -> tuple[List[float], List[float]]:
    """Extract easting/northing lists in the GeoDataFrame CRS (expected UTM for NEON AOP)."""
    if gdf.crs is None:
        raise ValueError("GeoDataFrame has no CRS; cannot derive tile coordinates.")
    xs = gdf.geometry.x.astype(float).tolist()
    ys = gdf.geometry.y.astype(float).tolist()
    return xs, ys


def by_tile_aop(
    dpid: str,
    site: str,
    year: int,
    easting: Sequence[float],
    northing: Sequence[float],
    savepath: str,
    buffer: int = 0,
    token: str | None = None,
    check_size: bool = False,
) -> None:
    """Download all AOP tiles intersecting the given coordinates (wraps ``neonutilities.by_tile_aop``)."""
    _require_neonutilities()
    os.makedirs(savepath, exist_ok=True)
    nu.by_tile_aop(
        dpid=dpid,
        site=site,
        year=year,
        easting=list(easting),
        northing=list(northing),
        buffer=buffer,
        savepath=savepath,
        token=token,
        check_size=check_size,
    )


def download_products_for_points(
    gdf: gpd.GeoDataFrame,
    site: str,
    years: Iterable[int],
    save_root: str,
    products: dict[str, str],
    buffer: int = 0,
    token: str | None = None,
    check_size: bool = False,
) -> None:
    """For each year and each product id, run ``by_tile_aop`` using all stem coordinates in ``gdf``."""
    easting, northing = easting_northing_from_geodataframe(gdf)
    for year in years:
        for label, dpid in products.items():
            target = os.path.join(save_root, site, str(year), label)
            by_tile_aop(
                dpid=dpid,
                site=site,
                year=int(year),
                easting=easting,
                northing=northing,
                savepath=target,
                buffer=buffer,
                token=token,
                check_size=check_size,
            )
