import os
from comet_ml import Experiment
from datetime import datetime
import glob
import geopandas as gpd
import pandas as pd
from random import randint
from time import sleep
import tensorflow as tf
from tensorflow import keras as tfk

from DeepTreeAttention.trees import AttentionModel
from DeepTreeAttention import __file__ as ROOT
from DeepTreeAttention.models.layers import WeightedSum
from DeepTreeAttention.generators import neighbors

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
model.create()
model.ensemble_model = tfk.models.load_model("{}/Ensemble.h5".format(model.config["neighbors"]["model_dir"]), custom_objects={"WeightedSum":WeightedSum})
hyperspectral_pool = glob.glob(model.config["hyperspectral_sensor_pool"], recursive=True)

#Load field data
ROOT = os.path.dirname(os.path.dirname(ROOT))
train = gpd.read_file("{}/data/processed/train.shp".format(ROOT))
test = gpd.read_file("{}/data/processed/test.shp".format(ROOT))


site_classes_file =  "{}/data/processed/site_class_labels.csv".format(ROOT)     
site_classdf  = pd.read_csv(site_classes_file)
site_label_dict = site_classdf.set_index("siteID").label.to_dict()

domain_classes_file =  "{}/data/processed/domain_class_labels.csv".format(ROOT)     
domain_classdf  = pd.read_csv(domain_classes_file)
domain_label_dict = domain_classdf.set_index("domainID").label.to_dict()

#client = start_cluster.start(cpus=2)

#Train - unique ids
train_ids = train.individual.unique()
train_dict = {}
for x in train_ids:
    train_dict[x] = neighbors.extract_features(df=train, x=x, model=model, hyperspectral_pool=hyperspectral_pool, site_label_dict=site_label_dict, domain_label_dict=domain_label_dict)
    
#Test - unique ids
test_ids = test.individual.unique()
test_dict = {}
for x in test_ids:
    test_dict[x] = neighbors.extract_features(df=test, x=x, model=model, hyperspectral_pool=hyperspectral_pool, site_label_dict=site_label_dict, domain_label_dict=domain_label_dict)

