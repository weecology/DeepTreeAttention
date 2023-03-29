import rasterio as rio
import glob
import os
from src import neon_paths
from src import start_cluster
from distributed import wait

client = start_cluster.start(cpus=30)
rgb_pool = glob.glob("/orange/ewhite/NeonData/*/DP3.30010.001/**/Camera/**/*.tif", recursive=True)
fils = glob.glob("/orange/ewhite/b.weinstein/DeepTreeAttention/Hyperspectral_tifs/year/*.tif")

def delete_if_error(f):
     try:
          img = rio.open(f).read()
     except:
          print(f)
          os.remove(f)     

futures = client.map(delete_if_error, fils)
wait(futures)
     
         