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
parser.add_argument("-branch")
parser.add_argument("-site")
parser.add_argument("-m",
                    required=False,
                    default=None)

args = parser.parse_args()
site = args.site

git_commit = subprocess.check_output("git log --pretty=format:'%H' -n 1", shell=True).decode()
git_branch = args.branch

config = data.read_config("config.yml")

if args.m:
    hot_config_fix = json.loads(args.m)
    for key, value in hot_config_fix.items():
        print("setting config {} to {}".format(key, value))
        config[key] = value

if config["use_data_commit"] is None:
    comet_logger = CometLogger(project_name="DeepTreeAttention2", workspace=config["comet_workspace"], auto_output_logging="simple") 
    comet_logger.experiment.add_tag("data_generation")
    client = start_cluster.start(cpus=5, mem_size="10GB")    
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
        site=site,
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
