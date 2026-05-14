import geopandas as gpd
import pandas as pd
import pytest
import rasterio
from shapely.geometry import box

from src.pipelines import osbs_mortality


def test_parse_tile_id_from_neon_rgb_filename():
    path = "/tmp/2025_D03_OSBS_DP3_404000_3284000_image_crop_2025.tif"
    assert osbs_mortality.parse_tile_id(path) == "404000_3284000"


def test_parse_tile_id_requires_neon_rgb_pattern():
    with pytest.raises(ValueError, match="Could not parse"):
        osbs_mortality.parse_tile_id("/tmp/not_a_neon_tile.tif")


def test_summarize_tile_crowns_counts_alive_dead_and_unknown():
    crowns = gpd.GeoDataFrame(
        {
            "dead_label": [1, 1, 0, None, "dead"],
            "dead_score": [0.96, 0.4, 0.99, 0.99, None],
        },
        geometry=[box(i, 0, i + 1, 1) for i in range(5)],
        crs="EPSG:32617",
    )

    summary = osbs_mortality.summarize_tile_crowns(
        crowns,
        tile_path="/tmp/2017_D03_OSBS_DP3_404000_3284000_image.tif",
        year=2017,
        dead_threshold=0.95,
    )

    assert summary["tile_id"] == "404000_3284000"
    assert summary["total_count"] == 5
    assert summary["dead_count"] == 2
    assert summary["alive_count"] == 2
    assert summary["unknown_count"] == 1
    assert summary["dead_fraction"] == 0.4


def test_compare_year_counts_flags_high_mortality():
    counts = gpd.GeoDataFrame(
        [
            {
                "tile_id": "404000_3284000",
                "year": 2017,
                "total_count": 10,
                "alive_count": 8,
                "dead_count": 2,
                "unknown_count": 0,
                "dead_fraction": 0.2,
            },
            {
                "tile_id": "404000_3284000",
                "year": 2025,
                "total_count": 11,
                "alive_count": 5,
                "dead_count": 6,
                "unknown_count": 0,
                "dead_fraction": 6 / 11,
            },
        ],
        geometry=[box(404000, 3284000, 405000, 3285000)] * 2,
        crs="EPSG:32617",
    )

    changes = osbs_mortality.compare_year_counts(
        counts,
        baseline_year=2017,
        comparison_year=2025,
        high_mortality_dead_change=3,
    )

    assert changes.shape[0] == 1
    row = changes.iloc[0]
    assert row["dead_count_change"] == 4
    assert row["alive_count_change"] == -3
    assert bool(row["high_mortality"]) is True


def test_write_metric_raster_uses_tile_values(tmp_path):
    tiles = gpd.GeoDataFrame(
        {
            "tile_id": ["404000_3284000", "405000_3284000"],
            "dead_count": [6, 2],
        },
        geometry=[
            box(404000, 3284000, 405000, 3285000),
            box(405000, 3284000, 406000, 3285000),
        ],
        crs="EPSG:32617",
    )
    output = tmp_path / "dead_count.tif"

    osbs_mortality.write_metric_raster(
        tiles,
        metric="dead_count",
        output_path=str(output),
        resolution=1000,
    )

    with rasterio.open(output) as src:
        data = src.read(1)
        assert src.width == 2
        assert src.height == 1
        assert src.tags()["metric"] == "dead_count"
        assert set(pd.Series(data.flatten()).dropna().astype(int)) == {2, 6}
