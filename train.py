# Train
import sys
from src import train, data

if __name__ == "__main__":
    #Get branch name for the comet tag
    git_branch=sys.argv[1]
    git_commit=sys.argv[2]    
    config = data.read_config("config.yml")
    for site in ["BART","BLAN", "BONA","CLBJ", "DEJU", "DELA", "GRSM", "HARV", "JERC",
                  "LENO", "MLBS", "MOAB", "NIWO" ,"OSBS","RMNP","SCBI","SERC","SJER","SOAP",
                 "STEI","TALL","TEAK","TREE","UKFS","UNDE","WREF","YELL"]:
        train.main(git_branch, git_commit, config, site)
