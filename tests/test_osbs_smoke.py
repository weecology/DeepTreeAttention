import json
import os
import shutil

import geopandas as gpd
import pandas as pd
import pytest
import rasterio
import yaml
from shapely.geometry import box

from src import osbs_smoke


def _write_smoke_manifest(manifest_path, tile_glob, aoi_path, output_dir, config_path):
    manifest = {
        "site": "OSBS",
        "year": 2025,
        "aoi_path": aoi_path,
        "tile_limit": 1,
        "io": {
            "config_path": config_path,
            "tile_glob": tile_glob,
            "output_dir": output_dir,
        },
        "detection": {
            "mode": "placeholder",
            "dead_model_path": None,
        },
        "classification": {
            "mode": "placeholder",
            "model_aliases": ["species_ensemble_placeholder"],
        },
    }
    with open(manifest_path, "w", encoding="utf-8") as outfile:
        yaml.safe_dump(manifest, outfile)


def test_manifest_requires_expected_top_level_keys(tmp_path):
    bad_manifest = tmp_path / "bad_manifest.yaml"
    bad_manifest.write_text("site: OSBS\n", encoding="utf-8")

    with pytest.raises(ValueError):
        osbs_smoke.load_manifest(str(bad_manifest))


def test_smoke_pipeline_creates_output_contract(tmp_path, ROOT):
    source_tile = os.path.join(ROOT, "tests", "data", "2019_D01_HARV_DP3_726000_4699000_image_crop_2019.tif")
    tile_dir = tmp_path / "OSBS" / "2025" / "Camera"
    tile_dir.mkdir(parents=True, exist_ok=True)
    target_tile = tile_dir / "2025_D03_OSBS_DP3_726000_4699000_image_crop_2025.tif"
    shutil.copy(source_tile, target_tile)

    with rasterio.open(target_tile) as src:
        tile_poly = box(*src.bounds)
        tile_crs = src.crs

    aoi = gpd.GeoDataFrame({"site": ["OSBS"]}, geometry=[tile_poly], crs=tile_crs)
    aoi_path = tmp_path / "aoi.geojson"
    aoi.to_file(aoi_path, driver="GeoJSON")

    manifest_path = tmp_path / "manifest.yaml"
    _write_smoke_manifest(
        manifest_path=manifest_path,
        tile_glob=str(tmp_path / "OSBS" / "**" / "Camera" / "*.tif"),
        aoi_path=str(aoi_path),
        output_dir=str(tmp_path / "outputs"),
        config_path=os.path.join(ROOT, "config.yml"),
    )

    context = osbs_smoke.run_smoke_pipeline(str(manifest_path))

    assert context.tiles_considered == 1
    assert len(context.tiles_selected) == 1
    assert len(context.crowns_output) == 1
    assert len(context.predictions_output) == 1

    metadata_path = tmp_path / "outputs" / "run_metadata.json"
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["tile_limit"] == 1
    assert metadata["detection_mode"] == "placeholder"
    assert metadata["classification_mode"] == "placeholder"

    predictions = pd.read_csv(context.predictions_output[0])
    assert set(predictions.columns) == {"individual", "ensembleTaxonID", "ens_score"}
