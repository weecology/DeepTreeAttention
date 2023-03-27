import rasterio as rio
import glob
import os
from src import neon_paths

hyperspectral_pool = glob.glob("/orange/ewhite/NeonData/*/DP3.30006.001/**/Reflectance/*.h5", recursive=True)
rgb_pool = glob.glob("/orange/ewhite/NeonData/*/DP3.30010.001/**/Camera/**/*.tif", recursive=True)

#def make_new_raster(f, hyperspectral_pool=hyperspectral_pool,rgb_pool=rgb_pool):
     #geo_index = "_".join(os.path.splitext(os.path.basename(f))[0].split("_")[3:5])
     #hsi_path = neon_paths.find_sensor_path(hyperspectral_pool, bounds=None, geo_index=geo_index)
     #print("check HSI path {}".format(hsi_path))
     #os.remove(hsi_path)
     ##Make next tile
     #hyperspectral_pool = [x for x in hyperspectral_pool if not x == hsi_path]
     #filename = neon_paths.lookup_and_convert(rgb_pool, hyperspectral_pool, savedir="/orange/idtrees-collab/Hyperspectral_tifs/previous_errors", geo_index=geo_index)
     #print("created new file {}".format(filename))

fils = glob.glob("/orange/ewhite/b.weinstein/DeepTreeAttention/Hyperspectral_tifs/year/*.tif")
for f in fils:
     try:
          img = rio.open(f).read()
     except:
          print(f)
          os.remove(f)

     
         