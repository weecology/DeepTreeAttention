import tarfile
import glob
import tempfile
import os
from src import utils
from src import start_cluster
from src.model_list import species_model_paths
import traceback
from distributed import wait
import numpy as np

client = start_cluster.start(cpus=150, mem_size="20GB")

tars = glob.glob("/blue/ewhite/b.weinstein/DeepTreeAttention/results/site_crops/*/tar/*")

def check_tar(tar):
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            with tarfile.open(tar, 'r') as archive:
                archive.extractall(temp_dir)
                npys = glob.glob("{}/*.npy".format(temp_dir))
                for npy in npys:
                    image = np.load(npy)
                    if (image == -9999).any():
                        raise ValueError("Input image path {} had NA value of -9999".format(npy))
         
    except (tarfile.ReadError, ValueError):
        basename = os.path.splitext(os.path.basename(tar))[0].split(".")[0]
        site = basename.split("_")[1]
        try:
            os.remove(tar)
        except:
            pass
        shp_path = "/blue/ewhite/b.weinstein/DeepTreeAttention/results/site_crops/{}/shp/{}.shp".format(site, basename)
        try:
            os.remove(shp_path)
        except:
            pass
        print("deleting {}".format(tar))
        print("deleting {}".format(shp_path))

for tar in tars:
    futures = client.map(check_tar, tars)

wait(futures)


        