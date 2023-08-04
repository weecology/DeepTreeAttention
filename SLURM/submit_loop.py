import pandas as pd
import subprocess
import os

params = pd.read_csv("params.csv",quotechar="'")
for index, row in params.iterrows():
    print(row.mydict)
    call = "sbatch SLURM/loop.sh {} {} '{}'".format(row.branch, row.siteID, row.mydict)
    subprocess.call(call, shell=True)