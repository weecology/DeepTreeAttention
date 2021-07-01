#test patches
import os
import glob
from src import patches
from src import __file__ as ROOT

ROOT = os.path.dirname(os.path.dirname(ROOT))

def test_patches():
    rgb_pool = glob.glob("{}/tests/data/*.tif".format(ROOT))
    patches.crown_to_pixel(path="{}/tests/data/crown.shp".format(ROOT), rgb_pool=rgb_pool)