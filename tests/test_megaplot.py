#test megaplot
import pytest
from src import megaplot
from src import generate
import os

ROOT = os.path.dirname(os.path.dirname(megaplot.__file__))

def test_read_files():
    formatted_data = megaplot.read_files(directory="{}/tests/data/MegaPlots/".format(ROOT))
    assert all([x in formatted_data["OSBS"].columns for x in ["individualID","plotID","siteID","taxonID"]])

def test_points_to_crowns(tmpdir):
    formatted_data = megaplot.read_files(directory="{}/tests/data/MegaPlots/".format(ROOT))
    formatted_data["OSBS"].to_file("{}/OSBS_points.shp".format(tmpdir))
    crowns = generate.points_to_crowns(
        field_data="{}/OSBS_points.shp".format(tmpdir),
        rgb_dir="{}/tests/data/MegaPlots/*.tif".format(ROOT),
        savedir=tmpdir,
        raw_box_savedir=tmpdir)
    assert all([x in crowns.columns for x in ["plotID","taxonID","individual"]])
