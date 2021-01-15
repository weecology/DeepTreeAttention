import os
from comet_ml import Experiment
from datetime import datetime
import pandas as pd
from random import randint
from time import sleep
import tensorflow as tf
from tensorflow import keras as tfk

from DeepTreeAttention.trees import AttentionModel
from DeepTreeAttention.models.layers import WeightedSum
from DeepTreeAttention.models import neighbors_model
from DeepTreeAttention.callbacks import callbacks

sleep(randint(0,5))
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = "{}/{}".format("/orange/idtrees-collab/DeepTreeAttention/snapshots/",timestamp)
os.mkdir(save_dir)

experiment = Experiment(project_name="neontrees", workspace="bw4sz")
experiment.add_tag("neighbors")

#Create output folder
experiment.log_parameter("timestamp",timestamp)
experiment.log_parameter("log_dir",save_dir)

#Create a class and run
model = AttentionModel(config="/home/b.weinstein/DeepTreeAttention/conf/tree_config.yml", log_dir=save_dir)
model.create()
model.ensemble_model = tfk.models.load_model("{}/Ensemble.h5".format(model.config["neighbors"]["model_dir"]), custom_objects={"WeightedSum":WeightedSum})
model.read_data("neighbors")

experiment.log_parameters(model.config["neighbors"])

neighbor = neighbors_model.create(ensemble_model = model.ensemble_model, k_neighbors=model.config["neighbors"]["k_neighbors"], classes=model.classes, freeze=model.config["neighbors"]["freeze"])

labeldf = pd.read_csv(model.classes_file)
label_names = list(labeldf.taxonID.values)

callback_list = callbacks.create(
    experiment = experiment,
    train_data = model.train_split,
    validation_data = model.val_split,
    train_shp = model.train_shp,
    log_dir=save_dir,
    label_names=label_names,
    submodel=False)

neighbor.fit(
    model.train_split,
    epochs=model.config["train"]["ensemble"]["epochs"],
    validation_data=model.val_split,
    callbacks=callback_list)

#save
neighbor.save("{}/neighbors.h5".format(save_dir))

predicted_shp = model.predict(model = model.neighbor_model)
predicted_shp.to_file("{}/prediction.shp".format(save_dir))
experiment.log_asset("{}/prediction.shp".format(save_dir))
experiment.log_asset("{}/prediction.dbf".format(save_dir))
experiment.log_asset("{}/prediction.shx".format(save_dir))
experiment.log_asset("{}/prediction.cpg".format(save_dir))

estimate_a = model.neighbor_model.get_layer("ensemble_add_bias").get_weights()
experiment.log_metric(name="target_versus_context_weight", value=estimate_a[0][0])