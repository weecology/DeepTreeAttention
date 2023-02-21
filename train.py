# Train
import sys
from src import train, data, start_cluster
import torch
import gc
import os
from pytorch_lightning.loggers import CometLogger
import traceback

git_branch=sys.argv[1]
git_commit=sys.argv[2] 
config = data.read_config("config.yml")
#sites = ["BART","BLAN", "BONA","CLBJ", "DEJU", "DELA", "GRSM", "HARV", "JERC",
              #"LENO", "MLBS", "MOAB", "NIWO" ,"OSBS","RMNP","SCBI","SERC","SJER","SOAP",
             #"STEI","TALL","TEAK","TREE","UKFS","UNDE","WREF","YELL"]

sites = [["OSBS","JERC","TALL"], ["TEAK"],["CLBJ"]]

#if config["use_data_commit"] is None:
    #comet_logger = CometLogger(project_name="DeepTreeAttention2", workspace=config["comet_workspace"], auto_output_logging="simple") 
    #comet_logger.experiment.add_tag("per_site_multi_stage_data_generation")
    #client = start_cluster.start(cpus=50, mem_size="4GB")    
    #ROOT = os.path.dirname(os.path.dirname(data.__file__))    
    #crop_dir = os.path.join(config["data_dir"], comet_logger.experiment.get_key())
    #os.mkdir(crop_dir)    
    #data_module = data.TreeData(
        #csv_file="{}/data/raw/neon_vst_data_2022.csv".format(ROOT),
        #data_dir=crop_dir,
        #experiment_id=git_commit,
        #config=config,
        #client=client,
        #site="all",
        #comet_logger=comet_logger)
    #config["use_data_commit"] = comet_logger.experiment.id 
    
for site in sites:
    try:
        train.main(site=site, config=config, git_branch=git_branch, git_commit=git_commit)
        torch.cuda.empty_cache() 
        gc.collect()            
    except:
        print("{} failed with {}".format(site, traceback.print_exc()))
