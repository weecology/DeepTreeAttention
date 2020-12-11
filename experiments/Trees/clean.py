from time import sleep
from random import randint
from datetime import datetime
import os
from comet_ml import Experiment
from DeepTreeAttention.trees import AttentionModel

sleep(randint(0,20))
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = "{}/{}".format("/orange/idtrees-collab/DeepTreeAttention/snapshots/",timestamp)
os.mkdir(save_dir)

experiment = Experiment(project_name="neontrees", workspace="bw4sz")
experiment.add_tag("Cleaning")

#Create output folder
experiment.log_parameter("timestamp",timestamp)
experiment.log_parameter("log_dir",save_dir)

#Create a class and run
model = AttentionModel(config="/home/b.weinstein/DeepTreeAttention/conf/tree_config.yml", log_dir=save_dir)
error_df = model.find_outliers()
error_df.to_csv("{}/outliers.csv".format(save_dir))