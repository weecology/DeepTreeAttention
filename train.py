# Train
import sys
import glob
import pandas as pd
from src import train, data

if __name__ == "__main__":
    #Get branch name for the comet tag
    git_branch=sys.argv[1]
    git_commit=sys.argv[2]    
    config = data.read_config("config.yml")
    
    if config["all_site_pretrain"]:
        files = glob.glob("{}/{}/*{}*".format(config["data_dir"],config["use_data_commit"],config["train_test_commit"]))
        files = [x for x in files if "train" in x]
        dfs = [pd.read_csv(x) for x in files]
        df = pd.concat(dfs)
        traindf = df.reset_index(drop=True)
        testdf.to_csv("{}/{}/train_{}_pretrain.csv".format(config["data_dir"],config["use_data_commit"], config["train_test_commit"]))
        
        files = glob.glob("{}/{}/*{}*".format(config["data_dir"],config["use_data_commit"],config["train_test_commit"]))
        files = [x for x in files if "test" in x]
        dfs = [pd.read_csv(x) for x in files]
        df = pd.concat(dfs)
        testdf = df.reset_index(drop=True)
        testdf.to_csv("{}/{}/test_{}_pretrain.csv".format(config["data_dir"],config["use_data_commit"], config["train_test_commit"]))
        
        site = "pretrain"        
        train.main(git_branch, git_commit, config, site)
        config["pretrain_state_dict"] = "{}_{}_pretrain_state_dict.pl".format(config["snapshot_dir"],config["train_test_commit"])
        
    for site in ["BART","BLAN", "BONA","CLBJ", "DEJU", "DELA", "GRSM", "HARV", "JERC",
                  "LENO", "MLBS", "MOAB", "NIWO" ,"OSBS","RMNP","SCBI","SERC","SJER","SOAP",
                 "STEI","TALL","TEAK","TREE","UKFS","UNDE","WREF","YELL"]:
        train.main(git_branch, git_commit, config, site)
