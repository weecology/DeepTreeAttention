"""Local smoke pipeline scaffold for OSBS single-tile prediction updates."""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import geopandas as gpd
import pandas as pd
import rasterio
import yaml
from shapely.geometry import box

from src import utils


@dataclass
class SmokeRunContext:
    """Runtime context for the smoke pipeline execution."""

    manifest_path: str
    output_dir: str
    tiles_considered: int
    tiles_selected: list[str]
    crowns_output: list[str]
    predictions_output: list[str]


def _resolve_path(path: str, base_dir: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(base_dir, path))


def load_manifest(manifest_path: str) -> dict[str, Any]:
    with open(manifest_path, "r", encoding="utf-8") as infile:
        manifest = yaml.safe_load(infile)

    required_keys = ["site", "year", "aoi_path", "tile_limit", "io", "detection", "classification"]
    missing = [key for key in required_keys if key not in manifest]
    if missing:
        raise ValueError(f"Manifest missing required keys: {missing}")

    return manifest


def load_aoi_polygon(aoi_path: str) -> gpd.GeoDataFrame:
    aoi = gpd.read_file(aoi_path)
    if aoi.empty:
        raise ValueError(f"AOI file is empty: {aoi_path}")
    return aoi


def select_osbs_tiles(manifest: dict[str, Any], aoi: gpd.GeoDataFrame, base_dir: str) -> tuple[list[str], int]:
    tile_glob = manifest["io"]["tile_glob"]
    tile_glob = _resolve_path(tile_glob, base_dir)
    candidates = glob.glob(tile_glob, recursive=True)

    site = manifest["site"]
    year = str(manifest["year"])
    filtered = [path for path in candidates if site in path and f"/{year}/" in path]
    aoi_union = aoi.geometry.union_all()

    selected: list[str] = []
    for tile_path in filtered:
        with rasterio.open(tile_path) as src:
            tile_geom = gpd.GeoSeries([box(*src.bounds)], crs=src.crs)
        tile_geom = tile_geom.to_crs(aoi.crs)
        if tile_geom.iloc[0].intersects(aoi_union):
            selected.append(tile_path)
        if len(selected) >= int(manifest["tile_limit"]):
            break

    return selected, len(filtered)


def _run_detection_stage(tile_path: str, config: dict[str, Any], manifest: dict[str, Any]) -> gpd.GeoDataFrame:
    mode = manifest["detection"].get("mode", "existing")
    dead_model_path = manifest["detection"].get("dead_model_path")
    dead_model_path = dead_model_path if dead_model_path else None

    if mode == "placeholder":
        return gpd.GeoDataFrame({"individual": [], "tile": []}, geometry=[], crs="EPSG:32617")
    if mode == "existing":
        from src import predict as predict_module

        crowns = predict_module.find_crowns(rgb_path=tile_path, config=config, dead_model_path=dead_model_path)
        if crowns is None:
            return gpd.GeoDataFrame({"individual": [], "tile": []}, geometry=[], crs="EPSG:32617")
        return crowns

    raise ValueError(f"Unsupported detection.mode: {mode}")


def _run_classification_stage(crowns: gpd.GeoDataFrame, manifest: dict[str, Any]) -> pd.DataFrame:
    mode = manifest["classification"].get("mode", "placeholder")
    if mode != "placeholder":
        raise ValueError("Smoke scaffold currently supports only classification.mode=placeholder")

    if crowns.empty:
        return pd.DataFrame(columns=["individual", "ensembleTaxonID", "ens_score"])

    prediction_df = pd.DataFrame(
        {
            "individual": crowns["individual"].values,
            "ensembleTaxonID": ["UNRESOLVED_MODEL_ON_HPC"] * crowns.shape[0],
            "ens_score": [None] * crowns.shape[0],
        }
    )
    return prediction_df


def run_smoke_pipeline(manifest_path: str) -> SmokeRunContext:
    manifest_path = os.path.abspath(manifest_path)
    base_dir = os.path.dirname(manifest_path)
    manifest = load_manifest(manifest_path)

    output_dir = _resolve_path(manifest["io"]["output_dir"], base_dir)
    os.makedirs(output_dir, exist_ok=True)

    aoi_path = _resolve_path(manifest["aoi_path"], base_dir)
    aoi = load_aoi_polygon(aoi_path)
    config_path = _resolve_path(manifest["io"]["config_path"], base_dir)
    config = utils.read_config(config_path=config_path)

    selected_tiles, candidate_count = select_osbs_tiles(manifest=manifest, aoi=aoi, base_dir=base_dir)

    crowns_output: list[str] = []
    predictions_output: list[str] = []
    for tile_path in selected_tiles:
        basename = os.path.splitext(os.path.basename(tile_path))[0]
        crowns_path = os.path.join(output_dir, f"{basename}_crowns.geojson")
        predictions_path = os.path.join(output_dir, f"{basename}_predictions.csv")

        crowns = _run_detection_stage(tile_path=tile_path, config=config, manifest=manifest)
        if not crowns.empty:
            aoi_union = aoi.geometry.union_all()
            crowns = crowns.to_crs(aoi.crs)
            crowns = crowns[crowns.geometry.intersects(aoi_union)]
        crowns.to_file(crowns_path, driver="GeoJSON")

        prediction_df = _run_classification_stage(crowns=crowns, manifest=manifest)
        prediction_df.to_csv(predictions_path, index=False)

        crowns_output.append(crowns_path)
        predictions_output.append(predictions_path)

    run_metadata_path = os.path.join(output_dir, "run_metadata.json")
    with open(run_metadata_path, "w", encoding="utf-8") as outfile:
        json.dump(
            {
                "manifest_path": manifest_path,
                "site": manifest["site"],
                "year": manifest["year"],
                "tile_limit": manifest["tile_limit"],
                "tiles_considered": candidate_count,
                "tiles_selected": selected_tiles,
                "created_at_utc": datetime.now(tz=timezone.utc).isoformat(),
                "detection_mode": manifest["detection"].get("mode", "existing"),
                "classification_mode": manifest["classification"].get("mode", "placeholder"),
                "outputs": {
                    "crowns": crowns_output,
                    "predictions": predictions_output,
                },
            },
            outfile,
            indent=2,
        )

    return SmokeRunContext(
        manifest_path=manifest_path,
        output_dir=output_dir,
        tiles_considered=candidate_count,
        tiles_selected=selected_tiles,
        crowns_output=crowns_output,
        predictions_output=predictions_output,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one-tile OSBS smoke prediction scaffold.")
    parser.add_argument("--manifest", required=True, help="Path to the smoke-run manifest yaml file.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    context = run_smoke_pipeline(manifest_path=args.manifest)
    print(
        json.dumps(
            {
                "tiles_considered": context.tiles_considered,
                "tiles_selected": context.tiles_selected,
                "output_dir": context.output_dir,
            },
            indent=2,
        )
    )

