"""Compare OSBS alive/dead tree counts between RGB years at NEON tile scale.

Run from the repository root after configuring ``osbs_mortality`` in
``config.yml``:

    python -m src.pipelines.osbs_mortality --config config.yml

The pipeline can download AOI-intersecting RGB tiles, run the existing
DeepForest detector plus Hugging Face alive/dead cropmodel, optionally run the
species model, then write per-tile count tables and GeoTIFF rasters.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import re
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pytorch_lightning import Trainer
from rasterio import features
from rasterio.transform import from_origin
from shapely.geometry import box

from src import utils
from src.models import multi_stage
from src.pipelines.osbs_inference import load_aoi, list_osbs_tiles_intersecting_aoi

RGB_DPID = "DP3.30010.001"
DEFAULT_SITE = "OSBS"
DEFAULT_AOP_EPSG = 32617
DEFAULT_TILE_SIZE_M = 1000
NO_DATA = -9999.0
_TILE_ID_RE = re.compile(r"_(?P<easting>\d{6})_(?P<northing>\d{7})_image(?:_|\.|$)")


def _resolve_path(path: str, anchor_dir: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(anchor_dir, path))


def parse_tile_id(path: str) -> str:
    """Return the NEON AOP easting/northing tile id from an RGB filename."""
    match = _TILE_ID_RE.search(os.path.basename(path))
    if not match:
        raise ValueError(f"Could not parse NEON tile id from filename: {path}")
    return f"{match.group('easting')}_{match.group('northing')}"


def tile_geometry_from_raster(path: str):
    with rasterio.open(path) as src:
        return box(*src.bounds), src.crs


def _require_mortality_block(config: dict[str, Any]) -> dict[str, Any]:
    block = config.get("osbs_mortality")
    if not block:
        raise ValueError(
            "config.yml must define an `osbs_mortality` mapping "
            "(years, aoi_path, results_root)."
        )
    for key in ("years", "aoi_path", "results_root"):
        if key not in block or block[key] in (None, "", []):
            raise ValueError(f"osbs_mortality.{key} is required")
    if len(block["years"]) < 2:
        raise ValueError("osbs_mortality.years must include at least two years")
    return block


def _as_dead_mask(crowns: gpd.GeoDataFrame, *, dead_threshold: float) -> pd.Series:
    if "dead_label" not in crowns.columns:
        return pd.Series(False, index=crowns.index)

    labels = crowns["dead_label"]
    numeric = pd.to_numeric(labels, errors="coerce")
    label_dead = numeric.eq(1)
    string_dead = labels.astype("string").str.lower().isin({"dead", "dead_tree", "1", "true"})
    label_dead = label_dead | string_dead.fillna(False)

    if "dead_score" not in crowns.columns:
        return label_dead

    scores = pd.to_numeric(crowns["dead_score"], errors="coerce")
    score_ok = scores.isna() | scores.ge(dead_threshold)
    return label_dead & score_ok


def summarize_tile_crowns(
    crowns: gpd.GeoDataFrame,
    *,
    tile_path: str,
    year: int,
    dead_threshold: float,
) -> dict[str, Any]:
    """Count alive/dead detections for one tile without linking individual trees."""
    dead_mask = _as_dead_mask(crowns, dead_threshold=dead_threshold)
    unknown_mask = (
        crowns["dead_label"].isna()
        if "dead_label" in crowns.columns
        else pd.Series(True, index=crowns.index)
    )
    total_count = int(len(crowns))
    dead_count = int(dead_mask.sum())
    unknown_count = int(unknown_mask.sum())
    alive_count = int(total_count - dead_count - unknown_count)
    if alive_count < 0:
        alive_count = 0

    return {
        "tile_id": parse_tile_id(tile_path),
        "year": int(year),
        "tile_path": tile_path,
        "total_count": total_count,
        "alive_count": alive_count,
        "dead_count": dead_count,
        "unknown_count": unknown_count,
        "dead_fraction": dead_count / total_count if total_count else 0.0,
    }


def compare_year_counts(
    counts: gpd.GeoDataFrame,
    *,
    baseline_year: int,
    comparison_year: int,
    high_mortality_dead_change: int,
) -> gpd.GeoDataFrame:
    """Build one row per tile comparing count changes between two years."""
    if counts.empty:
        return gpd.GeoDataFrame(geometry=[], crs=counts.crs)

    baseline = counts[counts["year"] == baseline_year].set_index("tile_id")
    comparison = counts[counts["year"] == comparison_year].set_index("tile_id")
    tile_ids = sorted(set(baseline.index) | set(comparison.index))
    rows: list[dict[str, Any]] = []
    geometries = []
    metrics = ("total_count", "alive_count", "dead_count", "unknown_count", "dead_fraction")

    for tile_id in tile_ids:
        row: dict[str, Any] = {
            "tile_id": tile_id,
            "baseline_year": int(baseline_year),
            "comparison_year": int(comparison_year),
        }
        base_row = baseline.loc[tile_id] if tile_id in baseline.index else None
        comp_row = comparison.loc[tile_id] if tile_id in comparison.index else None

        for metric in metrics:
            base_value = base_row[metric] if base_row is not None else np.nan
            comp_value = comp_row[metric] if comp_row is not None else np.nan
            row[f"{metric}_{baseline_year}"] = base_value
            row[f"{metric}_{comparison_year}"] = comp_value
            row[f"{metric}_change"] = (
                comp_value - base_value
                if not pd.isna(base_value) and not pd.isna(comp_value)
                else np.nan
            )

        row["high_mortality"] = bool(
            not pd.isna(row["dead_count_change"])
            and row["dead_count_change"] >= high_mortality_dead_change
        )
        rows.append(row)
        geometry_row = comp_row if comp_row is not None else base_row
        geometries.append(geometry_row.geometry)

    return gpd.GeoDataFrame(rows, geometry=geometries, crs=counts.crs)


def write_metric_raster(
    tile_metrics: gpd.GeoDataFrame,
    *,
    metric: str,
    output_path: str,
    resolution: int = DEFAULT_TILE_SIZE_M,
    nodata: float = NO_DATA,
) -> str:
    """Rasterize a tile-level metric, using one value for each NEON tile polygon."""
    if tile_metrics.empty:
        raise ValueError("Cannot write raster from empty tile metrics")
    if metric not in tile_metrics.columns:
        raise ValueError(f"Metric column not found: {metric}")

    valid = tile_metrics[~pd.isna(tile_metrics[metric])].copy()
    if valid.empty:
        raise ValueError(f"Metric column has no finite values: {metric}")

    minx, miny, maxx, maxy = valid.total_bounds
    width = max(1, int(math.ceil((maxx - minx) / resolution)))
    height = max(1, int(math.ceil((maxy - miny) / resolution)))
    transform = from_origin(minx, maxy, resolution, resolution)
    shapes = ((geom, float(value)) for geom, value in zip(valid.geometry, valid[metric]))
    array = features.rasterize(
        shapes=shapes,
        out_shape=(height, width),
        fill=nodata,
        transform=transform,
        dtype="float32",
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs=valid.crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(array, 1)
        dst.update_tags(metric=metric)
    return output_path


def _require_neonutilities():
    try:
        import neonutilities as nu
    except ImportError as exc:  # pragma: no cover - optional until downloads run
        raise ImportError(
            "The neonutilities package is required for osbs_mortality RGB downloads. "
            "Install project dependencies (e.g. uv sync) and retry."
        ) from exc
    return nu


def _call_by_tile_aop(nu, **kwargs) -> None:
    signature = inspect.signature(nu.by_tile_aop)
    supported = {k: v for k, v in kwargs.items() if k in signature.parameters}
    nu.by_tile_aop(**supported)


def _aop_extents_for_year(*, site: str, year: int, dpid: str = RGB_DPID) -> list[tuple[int, int]]:
    nu = _require_neonutilities()
    extents = nu.get_aop_tile_extents(dpid=dpid, site=site, year=year)
    return [(int(easting), int(northing)) for easting, northing in extents]


def select_aop_extents_intersecting_aoi(
    *,
    site: str,
    year: int,
    aoi: gpd.GeoDataFrame,
    aop_epsg: int = DEFAULT_AOP_EPSG,
    tile_size: int = DEFAULT_TILE_SIZE_M,
) -> list[tuple[int, int]]:
    """Return available AOP tile SW corners for ``year`` that intersect the AOI."""
    extents = _aop_extents_for_year(site=site, year=year)
    if not extents:
        return []

    aoi_utm = aoi.to_crs(epsg=aop_epsg)
    aoi_union = aoi_utm.geometry.union_all()
    selected: list[tuple[int, int]] = []
    for easting, northing in extents:
        geom = box(easting, northing, easting + tile_size, northing + tile_size)
        if geom.intersects(aoi_union):
            selected.append((easting, northing))
    return selected


def download_rgb_tiles(
    *,
    site: str,
    year: int,
    extents: Sequence[tuple[int, int]],
    save_root: str,
    token: str | None,
    check_size: bool = False,
    include_provisional: bool = True,
) -> str:
    """Download RGB data for a list of NEON AOP tile extents."""
    target = os.path.join(save_root, site, str(year), "rgb")
    os.makedirs(target, exist_ok=True)
    if not extents:
        return target

    easting, northing = zip(*extents)
    nu = _require_neonutilities()
    _call_by_tile_aop(
        nu,
        dpid=RGB_DPID,
        site=site,
        year=int(year),
        easting=list(easting),
        northing=list(northing),
        buffer=0,
        savepath=target,
        token=token,
        check_size=check_size,
        include_provisional=include_provisional,
    )
    return target


def _find_year_tiles(
    *,
    rgb_sensor_pool: str,
    site: str,
    year: int,
    aoi: gpd.GeoDataFrame,
    tile_limit: int | None,
) -> list[str]:
    tiles = list_osbs_tiles_intersecting_aoi(
        rgb_sensor_pool=rgb_sensor_pool,
        site=site,
        year=year,
        aoi=aoi,
    )
    if tile_limit is not None:
        tiles = tiles[:tile_limit]
    return sorted(tiles)


def _make_trainer(config: dict[str, Any], predict_limit_batches: int | None) -> Trainer:
    acc, dev = utils.trainer_accelerator_devices(config)
    kwargs: dict[str, Any] = {
        "accelerator": acc,
        "devices": dev,
        "logger": False,
        "enable_checkpointing": False,
    }
    if predict_limit_batches is not None:
        kwargs["limit_predict_batches"] = int(predict_limit_batches)
    return Trainer(**kwargs)


def _load_species_model(
    *,
    config: dict[str, Any],
    species_checkpoint: str | None,
    predict_limit_batches: int | None,
) -> tuple[Any, Trainer] | tuple[None, None]:
    if not species_checkpoint:
        return None, None
    if not os.path.isfile(species_checkpoint):
        raise FileNotFoundError(f"Species checkpoint not found: {species_checkpoint}")

    print("[osbs_mortality] loading MultiStage species checkpoint:", species_checkpoint)
    config["pretrained_state_dict"] = None
    model = multi_stage.MultiStage.load_from_checkpoint(species_checkpoint, config=config)
    return model, _make_trainer(config, predict_limit_batches)


def _write_tables(
    *,
    counts: gpd.GeoDataFrame,
    changes: gpd.GeoDataFrame,
    results_root: str,
) -> dict[str, str]:
    paths = {
        "counts_csv": os.path.join(results_root, "tile_alive_dead_counts.csv"),
        "counts_gpkg": os.path.join(results_root, "tile_alive_dead_counts.gpkg"),
        "changes_csv": os.path.join(results_root, "tile_alive_dead_changes.csv"),
        "changes_gpkg": os.path.join(results_root, "tile_alive_dead_changes.gpkg"),
    }
    os.makedirs(results_root, exist_ok=True)
    counts.drop(columns="geometry").to_csv(paths["counts_csv"], index=False)
    changes.drop(columns="geometry").to_csv(paths["changes_csv"], index=False)
    counts.to_file(paths["counts_gpkg"], driver="GPKG", layer="per_year_counts")
    changes.to_file(paths["changes_gpkg"], driver="GPKG", layer="year_change")
    return paths


def run_osbs_mortality(*, config_path: str) -> dict[str, Any]:
    config_path = os.path.abspath(config_path)
    anchor_dir = os.path.dirname(config_path)
    config = utils.read_config(config_path=config_path)
    block = _require_mortality_block(config)

    from src import predict as predict_module

    site = block.get("site", DEFAULT_SITE)
    years = [int(year) for year in block["years"]]
    baseline_year = int(block.get("baseline_year", years[0]))
    comparison_year = int(block.get("comparison_year", years[-1]))
    aoi_path = _resolve_path(str(block["aoi_path"]), anchor_dir)
    results_root = _resolve_path(str(block["results_root"]), anchor_dir)
    save_root = _resolve_path(
        str(block.get("save_root") or config.get("data_layout", {}).get("sensor_download_root", "data/external/neon_aop")),
        anchor_dir,
    )
    rgb_sensor_pool = block.get("rgb_sensor_pool") or config.get("rgb_sensor_pool")
    if rgb_sensor_pool is None:
        rgb_sensor_pool = os.path.join(save_root, site, "**", "rgb", "**", "*image.tif")
    elif not os.path.isabs(str(rgb_sensor_pool)):
        rgb_sensor_pool = _resolve_path(str(rgb_sensor_pool), anchor_dir)

    tile_limit = block.get("tile_limit")
    tile_limit = int(tile_limit) if tile_limit is not None else None
    dead_threshold = float(block.get("dead_threshold", config.get("dead_threshold", 0.95)))
    dead_model_path = block.get("dead_model_path") or None
    high_mortality_dead_change = int(block.get("high_mortality_dead_change", 10))
    tile_size = int(block.get("tile_size_m", DEFAULT_TILE_SIZE_M))
    aop_epsg = int(block.get("aop_epsg", DEFAULT_AOP_EPSG))
    predict_limit_batches = block.get("predict_limit_batches")
    predict_limit_batches = int(predict_limit_batches) if predict_limit_batches is not None else None
    download_rgb = bool(block.get("download_rgb", False))
    download_years: Iterable[int] = block.get("download_years") or years
    token = os.environ.get(block.get("token_env", "NEON_API_TOKEN"))

    inference_block = config.get("inference_osbs") or {}
    species_checkpoint = block.get("species_checkpoint", inference_block.get("species_checkpoint"))
    if species_checkpoint:
        species_checkpoint = _resolve_path(str(species_checkpoint), anchor_dir)
    run_species = bool(block.get("run_species", bool(species_checkpoint)))

    print("[osbs_mortality] config file:", config_path)
    print("[osbs_mortality] site:", site, "years:", years)
    print("[osbs_mortality] AOI:", aoi_path)
    print("[osbs_mortality] RGB pool:", rgb_sensor_pool)
    print("[osbs_mortality] results root:", results_root)
    print("[osbs_mortality] dead threshold:", dead_threshold)

    aoi = load_aoi(aoi_path)
    os.makedirs(results_root, exist_ok=True)
    download_log: list[dict[str, Any]] = []
    if download_rgb:
        for year in download_years:
            extents = select_aop_extents_intersecting_aoi(
                site=site,
                year=int(year),
                aoi=aoi,
                aop_epsg=aop_epsg,
                tile_size=tile_size,
            )
            print(f"[osbs_mortality] downloading {len(extents)} RGB tile(s) for {year}")
            target = download_rgb_tiles(
                site=site,
                year=int(year),
                extents=extents,
                save_root=save_root,
                token=token,
                check_size=bool(block.get("check_size", False)),
                include_provisional=bool(block.get("include_provisional", True)),
            )
            download_log.append({"year": int(year), "tiles": len(extents), "target": target})

    species_model, species_trainer = _load_species_model(
        config=config,
        species_checkpoint=species_checkpoint if run_species else None,
        predict_limit_batches=predict_limit_batches,
    )
    crowns_dir = os.path.join(results_root, "crowns")
    crops_root = os.path.join(results_root, "crops")
    pred_dir = os.path.join(results_root, "predictions")
    for path in (crowns_dir, crops_root, pred_dir):
        os.makedirs(path, exist_ok=True)

    count_records: list[dict[str, Any]] = []
    geometries = []
    output_crs = None
    run_log: list[dict[str, Any]] = []
    aoi_union = aoi.geometry.union_all()

    for year in years:
        tiles = _find_year_tiles(
            rgb_sensor_pool=str(rgb_sensor_pool),
            site=site,
            year=year,
            aoi=aoi,
            tile_limit=tile_limit,
        )
        print(f"[osbs_mortality] {year}: processing {len(tiles)} tile(s)")
        for index, rgb_path in enumerate(tiles, start=1):
            basename = os.path.splitext(os.path.basename(rgb_path))[0]
            print(f"\n[osbs_mortality] --- {year} tile {index}/{len(tiles)}: {basename} ---")
            try:
                crowns = predict_module.find_crowns(
                    rgb_path,
                    config,
                    dead_model_path=dead_model_path,
                )
            except Exception as exc:
                print(f"[osbs_mortality] ERROR find_crowns: {exc}")
                run_log.append({"year": year, "tile": rgb_path, "stage": "detection", "ok": False, "error": str(exc)})
                continue

            if crowns is None or crowns.empty:
                print("[osbs_mortality] no crowns; writing zero count for tile")
                crowns = gpd.GeoDataFrame(geometry=[], crs=output_crs)
            else:
                crowns = crowns.to_crs(aoi.crs)
                n_before = len(crowns)
                crowns = crowns[crowns.geometry.intersects(aoi_union)]
                print(f"[osbs_mortality] crowns after AOI clip: {len(crowns)} (was {n_before})")

            crowns_path = os.path.join(crowns_dir, f"{basename}.gpkg")
            crowns.to_file(crowns_path, driver="GPKG", layer="crowns")
            tile_geom, tile_crs = tile_geometry_from_raster(rgb_path)
            if output_crs is None:
                output_crs = tile_crs

            record = summarize_tile_crowns(
                crowns,
                tile_path=rgb_path,
                year=year,
                dead_threshold=dead_threshold,
            )
            count_records.append(record)
            geometries.append(tile_geom)

            if species_model is not None and species_trainer is not None and not crowns.empty:
                tile_config = dict(config)
                tile_config["prediction_crop_dir"] = os.path.join(crops_root, basename)
                os.makedirs(tile_config["prediction_crop_dir"], exist_ok=True)
                try:
                    crown_ann_path = predict_module.generate_prediction_crops(crowns, tile_config)
                    trees = predict_module.predict_tile(
                        crown_annotations=crown_ann_path,
                        m=species_model,
                        trainer=species_trainer,
                        filter_dead=bool(block.get("filter_dead", True)),
                        savedir=pred_dir,
                        config=tile_config,
                    )
                    run_log.append(
                        {
                            "year": year,
                            "tile": rgb_path,
                            "stage": "species",
                            "ok": True,
                            "crowns": len(crowns),
                            "trees": 0 if trees is None else len(trees),
                            "crowns_path": crowns_path,
                        }
                    )
                except Exception as exc:
                    print(f"[osbs_mortality] ERROR species prediction: {exc}")
                    run_log.append({"year": year, "tile": rgb_path, "stage": "species", "ok": False, "error": str(exc)})
            else:
                run_log.append(
                    {
                        "year": year,
                        "tile": rgb_path,
                        "stage": "alive_dead",
                        "ok": True,
                        "crowns": len(crowns),
                        "crowns_path": crowns_path,
                    }
                )

    counts = gpd.GeoDataFrame(count_records, geometry=geometries, crs=output_crs)
    changes = compare_year_counts(
        counts,
        baseline_year=baseline_year,
        comparison_year=comparison_year,
        high_mortality_dead_change=high_mortality_dead_change,
    )
    table_paths = _write_tables(counts=counts, changes=changes, results_root=results_root)

    raster_paths: dict[str, str] = {}
    comparison_counts = counts[counts["year"] == comparison_year]
    if not comparison_counts.empty:
        raster_paths["comparison_dead_count"] = write_metric_raster(
            comparison_counts,
            metric="dead_count",
            output_path=os.path.join(results_root, f"osbs_dead_count_{comparison_year}.tif"),
            resolution=tile_size,
        )
    if not changes.empty and "dead_count_change" in changes.columns:
        raster_paths["dead_count_change"] = write_metric_raster(
            changes,
            metric="dead_count_change",
            output_path=os.path.join(results_root, f"osbs_dead_count_change_{baseline_year}_{comparison_year}.tif"),
            resolution=tile_size,
        )

    metadata = {
        "created_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "config_path": config_path,
        "site": site,
        "years": years,
        "baseline_year": baseline_year,
        "comparison_year": comparison_year,
        "dead_threshold": dead_threshold,
        "high_mortality_dead_change": high_mortality_dead_change,
        "download_log": download_log,
        "per_tile": run_log,
        "tables": table_paths,
        "rasters": raster_paths,
    }
    meta_path = os.path.join(results_root, "run_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    metadata["metadata_path"] = meta_path

    print("[osbs_mortality] wrote:", table_paths)
    print("[osbs_mortality] wrote rasters:", raster_paths)
    print("[osbs_mortality] wrote metadata:", meta_path)
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run OSBS tile-level alive/dead mortality comparison.")
    parser.add_argument(
        "--config",
        default="config.yml",
        help="Path to config.yml (must define osbs_mortality).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_osbs_mortality(config_path=args.config)


if __name__ == "__main__":
    main()
