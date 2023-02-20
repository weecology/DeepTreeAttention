# Train
import sys
import glob
import pandas as pd
from src import train, data
import traceback
import torch
import gc
if __name__ == "__main__":
    #Get branch name for the comet tag
    git_branch=sys.argv[1]
    git_commit=sys.argv[2]    
    config = data.read_config("config.yml")
    original_commit = config["train_test_commit"]
    if config["all_site_pretrain"]:
        files = glob.glob("{}/{}/*{}*".format(config["data_dir"],config["use_data_commit"],config["train_test_commit"]))
        files = [x for x in files if "test" in x]
        dfs = [pd.read_csv(x) for x in files]
        df = pd.concat(dfs)
        testdf = df.reset_index(drop=True)
        testdf.to_csv("{}/{}/test_{}_all.csv".format(config["data_dir"],config["use_data_commit"], config["train_test_commit"]))
        site = "all"
        config["existing_test_csv"] = "{}/{}/test_{}_all.csv".format(config["data_dir"],config["use_data_commit"], config["train_test_commit"])
        config["train_test_commit"] = None
        comet_logger = train.main(git_branch, git_commit, config, site)
        config["pretrain_state_dict"] = "{}/{}_all_state_dict.pt".format(config["snapshot_dir"],comet_logger.experiment.id)
        config["train_test_commit"] = original_commit
        config["existing_test_csv"] = None
    
    #sites = ["BART","BLAN", "BONA","CLBJ", "DEJU", "DELA", "GRSM", "HARV", "JERC",
                  #"LENO", "MLBS", "MOAB", "NIWO" ,"OSBS","RMNP","SCBI","SERC","SJER","SOAP",
                 #"STEI","TALL","TEAK","TREE","UKFS","UNDE","WREF","YELL"]
    sites = [["OSBS","JERC","TALL"]]
    
    for site in sites:
        try:
            train.main(git_branch, git_commit, config, site)
            torch.cuda.empty_cache() 
            gc.collect()            
        except:
            print("{} failed with {}".format(site, traceback.print_exc()))
