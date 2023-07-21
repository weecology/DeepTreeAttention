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

git_branch = sys.argv[1]
git_commit = sys.argv[2] 
site = sys.argv[3]
config = data.read_config("config.yml")

#Sanitize string representation
site = site.replace("_",", ")
site = json.loads(site)

if config["use_data_commit"] is None:
    comet_logger = CometLogger(project_name="DeepTreeAttention2", workspace=config["comet_workspace"], auto_output_logging="simple") 
    comet_logger.experiment.add_tag("data_generation")
    client = start_cluster.start(cpus=100, mem_size="4GB")    
    ROOT = os.path.dirname(os.path.dirname(data.__file__))    
    crop_dir = os.path.join(config["data_dir"], comet_logger.experiment.get_key())
    os.mkdir(crop_dir) 
    config["crop_dir"] = crop_dir
    
    data_module = data.TreeData(
        csv_file="{}/data/raw/neon_vst_data_2022.csv".format(ROOT),
        data_dir=crop_dir,
        experiment_id=git_commit,
        config=config,
        client=client,
        create_train_test=True,
        site="pretrain",
        comet_logger=comet_logger)
    config["use_data_commit"] = comet_logger.experiment.id 
elif config["train_test_commit"] is None:
    client = start_cluster.start(cpus=2, mem_size="4GB")    
else:
    client = None
    
try:
    #Specific site instructions
    if "OSBS" in site:
        config["seperate_oak_model"] = True
        
    train.main(site=site, config=config, git_branch=git_branch, git_commit=git_commit, client=client)
    torch.cuda.empty_cache() 
    gc.collect()            
except:
    print("{} failed with {}".format(site, traceback.print_exc()))
