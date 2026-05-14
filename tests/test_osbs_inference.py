import os
import shutil

import geopandas as gpd
import pytest
import rasterio
from shapely.geometry import box

from src.pipelines import osbs_inference


def test_list_osbs_tiles_intersecting_aoi(tmp_path):
    source_tile = os.path.join(
        os.path.dirname(__file__), "data", "2019_D01_HARV_DP3_726000_4699000_image_crop_2019.tif"
    )
    tile_dir = tmp_path / "OSBS" / "2025" / "Camera"
    tile_dir.mkdir(parents=True)
    target = tile_dir / "2025_D03_OSBS_DP3_726000_4699000_image_crop_2025.tif"
    shutil.copy(source_tile, target)

    with rasterio.open(target) as src:
        tile_poly = box(*src.bounds)
        crs = src.crs

    aoi = gpd.GeoDataFrame({"id": [1]}, geometry=[tile_poly], crs=crs)
    pattern = str(tmp_path / "OSBS" / "**" / "Camera" / "*.tif")
    tiles = osbs_inference.list_osbs_tiles_intersecting_aoi(
        rgb_sensor_pool=pattern,
        site="OSBS",
        year=2025,
        aoi=aoi,
    )
    assert len(tiles) == 1
    assert os.path.basename(tiles[0]) == target.name


def test_run_osbs_inference_requires_block(tmp_path):
    cfg = tmp_path / "config.yml"
    cfg.write_text("comet_workspace: x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="inference_osbs"):
        osbs_inference.run_osbs_inference(config_path=str(cfg))


def test_run_osbs_inference_requires_keys(tmp_path):
    cfg = tmp_path / "config.yml"
    cfg.write_text("inference_osbs: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="year"):
        osbs_inference.run_osbs_inference(config_path=str(cfg))
