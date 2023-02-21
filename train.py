# Train
import sys
from src import train
import torch
import gc

git_branch=sys.argv[1]
git_commit=sys.argv[2] 
config = data.read_config("config.yml")
#sites = ["BART","BLAN", "BONA","CLBJ", "DEJU", "DELA", "GRSM", "HARV", "JERC",
              #"LENO", "MLBS", "MOAB", "NIWO" ,"OSBS","RMNP","SCBI","SERC","SJER","SOAP",
             #"STEI","TALL","TEAK","TREE","UKFS","UNDE","WREF","YELL"]
sites = [["OSBS","JERC","TALL"], "TEAK","CLBJ"]

for site in sites:
    try:
        train.main(site, config, git_branch, git_commit)
        torch.cuda.empty_cache() 
        gc.collect()            
    except:
        print("{} failed with {}".format(site, traceback.print_exc()))
