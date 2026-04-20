"""OSBS RGB tile inference: detect crowns, build HSI crops, run species MultiStage checkpoint.

All settings live under ``inference_osbs`` in ``config.yml``. Run:

    python -m src.pipelines.osbs_inference --config config.yml

Or from repo root after install:

    deeptree-infer-osbs --config config.yml
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from datetime import datetime, timezone
from typing import Any

import geopandas as gpd
import rasterio
from pytorch_lightning import Trainer
from shapely.geometry import box

from src import utils
from src.models import multi_stage


def _resolve_path(path: str, anchor_dir: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(anchor_dir, path))


def load_aoi(aoi_path: str) -> gpd.GeoDataFrame:
    aoi = gpd.read_file(aoi_path)
    if aoi.empty:
        raise ValueError(f"AOI is empty: {aoi_path}")
    return aoi


def list_osbs_tiles_intersecting_aoi(
    *,
    rgb_sensor_pool: str,
    site: str,
    year: int,
    aoi: gpd.GeoDataFrame,
) -> list[str]:
    """Return RGB tile paths for ``site`` and ``year`` that intersect the AOI polygon."""
    candidates = glob.glob(rgb_sensor_pool, recursive=True)
    year_token = f"/{year}/"
    filtered = [
        p
        for p in candidates
        if site in p and year_token in p and "neon-aop-products" not in p
    ]
    aoi_union = aoi.geometry.union_all()
    selected: list[str] = []
    for tile_path in filtered:
        with rasterio.open(tile_path) as src:
            tile_geom = gpd.GeoSeries([box(*src.bounds)], crs=src.crs)
        tile_geom = tile_geom.to_crs(aoi.crs)
        if tile_geom.iloc[0].intersects(aoi_union):
            selected.append(tile_path)
    return selected


def _require_inference_block(config: dict[str, Any]) -> dict[str, Any]:
    block = config.get("inference_osbs")
    if not block:
        raise ValueError(
            "config.yml must define an `inference_osbs` mapping "
            "(year, species_checkpoint, aoi_path, results_root)."
        )
    for key in ("year", "species_checkpoint", "aoi_path", "results_root"):
        if key not in block or block[key] in (None, ""):
            raise ValueError(f"inference_osbs.{key} is required")
    return block


def run_osbs_inference(*, config_path: str) -> None:
    config_path = os.path.abspath(config_path)
    anchor_dir = os.path.dirname(config_path)
    config = utils.read_config(config_path=config_path)
    inf = _require_inference_block(config)

    from src import predict as predict_module

    site = inf.get("site", "OSBS")
    year = int(inf["year"])
    species_ckpt = _resolve_path(str(inf["species_checkpoint"]), anchor_dir)
    aoi_path = _resolve_path(str(inf["aoi_path"]), anchor_dir)
    results_root = _resolve_path(str(inf["results_root"]), anchor_dir)
    tile_limit = inf.get("tile_limit")
    dead_model_path = inf.get("dead_model_path") or None
    filter_dead = bool(inf.get("filter_dead", True))

    print("[osbs_inference] config file:", config_path)
    print("[osbs_inference] site:", site, "year:", year)
    print("[osbs_inference] species checkpoint:", species_ckpt)
    print("[osbs_inference] AOI:", aoi_path)
    print("[osbs_inference] results root:", results_root)

    if not os.path.isfile(species_ckpt):
        raise FileNotFoundError(f"Species checkpoint not found: {species_ckpt}")

    aoi = load_aoi(aoi_path)
    tiles = list_osbs_tiles_intersecting_aoi(
        rgb_sensor_pool=config["rgb_sensor_pool"],
        site=site,
        year=year,
        aoi=aoi,
    )
    print(f"[osbs_inference] tiles intersecting AOI (before limit): {len(tiles)}")
    if tile_limit is not None:
        tiles = tiles[: int(tile_limit)]
        print(f"[osbs_inference] applied tile_limit={tile_limit}; processing {len(tiles)} tile(s)")

    crowns_dir = os.path.join(results_root, "crowns")
    pred_dir = os.path.join(results_root, "predictions")
    os.makedirs(crowns_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    print("[osbs_inference] loading MultiStage once from checkpoint …")
    config["pretrained_state_dict"] = None
    model = multi_stage.MultiStage.load_from_checkpoint(species_ckpt, config=config)
    trainer = Trainer(
        gpus=config["gpus"],
        logger=False,
        enable_checkpointing=False,
    )

    run_log: list[dict[str, Any]] = []
    for index, rgb_path in enumerate(tiles, start=1):
        basename = os.path.splitext(os.path.basename(rgb_path))[0]
        print(f"\n[osbs_inference] --- tile {index}/{len(tiles)}: {basename} ---")
        print("[osbs_inference] RGB path:", rgb_path)

        try:
            crowns = predict_module.find_crowns(
                rgb_path, config, dead_model_path=dead_model_path
            )
        except Exception as exc:
            print(f"[osbs_inference] ERROR find_crowns: {exc}")
            run_log.append({"tile": rgb_path, "stage": "detection", "ok": False, "error": str(exc)})
            continue

        if crowns is None or crowns.empty:
            print("[osbs_inference] no crowns; skipping tile")
            run_log.append({"tile": rgb_path, "stage": "detection", "ok": True, "crowns": 0})
            continue

        aoi_union = aoi.geometry.union_all()
        crowns = crowns.to_crs(aoi.crs)
        n_before = len(crowns)
        crowns = crowns[crowns.geometry.intersects(aoi_union)]
        print(f"[osbs_inference] crowns after AOI clip: {len(crowns)} (was {n_before})")
        if crowns.empty:
            print("[osbs_inference] no crowns inside AOI; skipping tile")
            run_log.append({"tile": rgb_path, "stage": "aoi_clip", "ok": True, "crowns": 0})
            continue

        crown_shp = os.path.join(crowns_dir, f"{basename}.shp")
        crowns.to_file(crown_shp)
        print("[osbs_inference] wrote crowns:", crown_shp)

        crop_dir = os.path.join(results_root, "crops", basename)
        os.makedirs(crop_dir, exist_ok=True)
        config["prediction_crop_dir"] = crop_dir

        try:
            print("[osbs_inference] generating prediction crops (HSI windows) …")
            crown_ann_path = predict_module.generate_prediction_crops(crowns, config)
        except Exception as exc:
            print(f"[osbs_inference] ERROR generate_prediction_crops: {exc}")
            run_log.append({"tile": rgb_path, "stage": "crops", "ok": False, "error": str(exc)})
            continue

        try:
            print("[osbs_inference] running species predict_tile …")
            trees = predict_module.predict_tile(
                crown_annotations=crown_ann_path,
                m=model,
                trainer=trainer,
                filter_dead=filter_dead,
                savedir=pred_dir,
                config=config,
            )
            n_trees = 0 if trees is None else len(trees)
            print(f"[osbs_inference] finished tile; trees written: {n_trees}")
            run_log.append(
                {"tile": rgb_path, "stage": "species", "ok": True, "trees": n_trees}
            )
        except Exception as exc:
            print(f"[osbs_inference] ERROR predict_tile: {exc}")
            run_log.append({"tile": rgb_path, "stage": "species", "ok": False, "error": str(exc)})

    meta_path = os.path.join(results_root, "run_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "created_at_utc": datetime.now(tz=timezone.utc).isoformat(),
                "config_path": config_path,
                "site": site,
                "year": year,
                "species_checkpoint": species_ckpt,
                "tiles_total": len(tiles),
                "per_tile": run_log,
            },
            f,
            indent=2,
        )
    print(f"\n[osbs_inference] wrote run metadata: {meta_path}")
    print("[osbs_inference] done.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run OSBS tile inference (detection + species).")
    p.add_argument(
        "--config",
        default="config.yml",
        help="Path to config.yml (must define inference_osbs).",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    run_osbs_inference(config_path=args.config)


if __name__ == "__main__":
    main()
