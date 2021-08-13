#Test data module
from src import data
from src import generate
import glob
import geopandas as gpd

import os
from distributed import Client
ROOT = os.path.dirname(os.path.dirname(data.__file__))


def test_TreeData(tmpdir):
    #dask client to test
    client = Client()
    #One site's worth of data
    csv_file = "{}/tests/data/sample_neon.csv".format(ROOT)
    #Turn of CHM filtering for the moment
    config = data.read_config(config_path="{}/config.yml".format(ROOT))
    config["min_CHM_height"] = None
    config["iterations"] = 1
    data_module = data.TreeData(config=config, data_dir="{}/tests/data".format(ROOT))
    data_module.setup(csv_file=csv_file, regenerate=True, client=client)
    
def test_TreeDataset(tmpdir):
    data_path = "{}/tests/data/crown.shp".format(ROOT)
    rgb_pool = glob.glob("{}/tests/data/*.tif".format(ROOT))
    gdf = gpd.read_file(data_path)
    annotations = generate.generate_crops(gdf=gdf, rgb_pool=rgb_pool, crop_save_dir=tmpdir)   
    annotations.to_csv("{}/train.csv".format(tmpdir))
    data_loader = data.TreeDataset(csv_file="{}/train.csv".format(tmpdir))
    assert len(data_loader) == annotations.shape[0]