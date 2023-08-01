# Train
import sys
from src import train, data, start_cluster
import torch
import gc
import os
from pytorch_lightning.loggers import CometLogger
import traceback
import copy
import json
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("git_branch")
parser.add_argument("git_commit")
parser.add_argument("site")
parser.add_argument("-m", "--mydict", action="store",
                    required=False, type=dict,
                    default={})


args = parser.parse_args()
hot_config = json.loads(args.mydict)
site = args.site

git_branch = subprocess.Popen('symbolic-ref HEAD 2>/dev/null || echo "(unnamed branch)")|cut -d/ -f3-)')
git_commit = subprocess.Popen("git log --pretty=format:'%H' -n 1")

config = data.read_config("config.yml")

for key, value in hot_config.items():
    config[key] = value

if config["use_data_commit"] is None:
    comet_logger = CometLogger(project_name="DeepTreeAttention2", workspace=config["comet_workspace"], auto_output_logging="simple") 
    comet_logger.experiment.add_tag("data_generation")
    client = start_cluster.start(cpus=100, mem_size="10GB")    
    ROOT = os.path.dirname(os.path.dirname(data.__file__))    
    crop_dir = os.path.join(config["data_dir"], comet_logger.experiment.get_key())
    os.mkdir(crop_dir) 
    config["crop_dir"] = crop_dir
    
    data_module = data.TreeData(
        csv_file="{}/data/raw/neon_vst_data_2022.csv".format(ROOT),
        data_dir=crop_dir,
        experiment_id=comet_logger.experiment.id,
        config=config,
        client=client,
        create_train_test=True,
        site="pretrain",
        comet_logger=comet_logger)
    config["use_data_commit"] = comet_logger.experiment.id 
elif config["train_test_commit"] is None:
    client = start_cluster.start(cpus=5, mem_size="4GB")    
else:
    client = None
    
try:
    #Specific site instructions
    if "OSBS" in site:
        config["seperate_oak_model"] = True
        
    train.main(
        site=site,
        config=config,
        git_branch=git_branch,
        git_commit=git_commit,
        client=client)
    torch.cuda.empty_cache() 
    gc.collect()            
except:
    print("{} failed with {}".format(site, traceback.print_exc()))
