from src import Hyperspectral
from src import neon_paths
from src import start_cluster
from distributed import wait
import glob
import h5py
import os

#rgb_pool = glob.glob("/orange/ewhite/NeonData/*/DP3.30010.001/**/Camera/**/*.tif", recursive=True)
#rgb_path = neon_paths.find_sensor_path(lookup_pool=rgb_pool, geo_index="328000_3603000")
#neon_paths.convert_h5(hyperspectral_h5_path="/orange/ewhite/NeonData/JORN/DP3.30006.001/2019/FullSite/D14/2019_JORN_3/L3/Spectrometer/Reflectance/NEON_D14_JORN_DP3_328000_3603000_reflectance.h5",
                      #rgb_path=rgb_path, savedir="/blue/ewhite/b.weinstein/")

hyperspectral_pool = glob.glob("/orange/ewhite/NeonData/*/DP3.30006.001/**/Reflectance/*.h5", recursive=True)
client = start_cluster.start(cpus=50)

def check_h5(path):
    try:
        h5py.File(path, 'r')
    except:
        #os.remove(path)
        return path

futures = client.map(check_h5, hyperspectral_pool)
wait(futures)

for x in futures:
    path = x.result()
    if path is not None:
        print(path)
        #os.remove(x.result())

