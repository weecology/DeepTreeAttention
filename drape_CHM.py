# Drape predictions
from src import CHM
from src.data import read_config
from src.utils import create_glob_lists
import glob
import geopandas as gpd

config = read_config("config.yml")
rgb_pool, h5_pool, hsi_pool, CHM_pool = create_glob_lists(config)

files = glob.glob("/blue/ewhite/b.weinstein/DeepTreeAttention/results/predictions/**/*.shp", recursive=True)

def drape(shp, config, CHM_pool):
    """Take in a predictions shapefile and extract CHM height"""
    shp = gpd.read_file(shp)
    # Create a dummy plotID
    shp["plotID"] = "same_plot"
    draped_shp = CHM.postprocess_CHM(shp, lookup_pool=CHM_pool)
    #draped_shp.to_file(shp)

for f in files:
    drape(f, config=config, CHM_pool=CHM_pool)
    print(f)