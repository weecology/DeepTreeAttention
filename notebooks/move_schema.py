import glob
from src import start_cluster
from src.utils import create_glob_lists
from src.data import read_config
from shutil import move
from distributed import wait
import os
client = start_cluster.start(cpus=50, mem_size="3GB")

config = read_config("config.yml")
rgb_pool, h5_pool, hsi_pool, CHM_pool = create_glob_lists(config)


def move_files(path):
    if not "neon-aop-products" in path:
       dest = "/".join(path.split("/")[:6]) + "/neon-aop-products/" + "/".join(path.split("/")[6:])
       print(dest)
       os.makedirs(os.path.dirname(dest), exist_ok=True)
       move(src=path,dst=dest)

futures = client.map(move_files,h5_pool)

for f in futures:
    f.result()

wait(futures)
