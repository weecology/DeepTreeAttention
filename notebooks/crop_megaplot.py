import geopandas as gpd
import rasterio as rio
from src import neon_paths
from src import data
import glob

config = data.read_config("config.yml")
rgb_pool = glob.glob(config["rgb_sensor_pool"], recursive=True)
dead_megaplot = gpd.read_file("/Users/benweinstein/Dropbox/Weecology/MegaPlots/OSBS/OSBS_dead_megaplot.shp")

for x in dead_megaplot.geometry:
    sensor_path = neon_paths.find_sensor_path(lookup_pool=rgb_pool, bounds=x.bounds)
    src = rio.open(sensor_path)
    box = src.read(window=rio.windows.from_bounds(left-1, bottom-1, right+1, top+1, transform=src.transform))
    