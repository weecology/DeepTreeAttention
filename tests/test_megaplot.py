#test megaplot
from src import megaplot
from src import generate
from src import utils

def test_read_files(config, ROOT):
    formatted_data = megaplot.read_files(directory="{}/tests/data/MegaPlots/".format(ROOT), config=config)
    assert all([x in formatted_data.columns for x in ["individualID","plotID","siteID","taxonID"]])

def test_points_to_crowns(tmpdir, config, ROOT):
    formatted_data = megaplot.read_files(directory="{}/tests/data/MegaPlots/".format(ROOT), config=config)
    formatted_data.to_file("{}/OSBS_points.shp".format(tmpdir))
    crowns = generate.points_to_crowns(
        field_data="{}/OSBS_points.shp".format(tmpdir),
        rgb_dir="{}/tests/data/MegaPlots/*.tif".format(ROOT),
        savedir=tmpdir,
        raw_box_savedir=tmpdir)
    assert all([x in crowns.columns for x in ["plotID","taxonID","individual"]])
