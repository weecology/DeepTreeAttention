import os
from random import randint
from time import sleep
from datetime import datetime

from comet_ml import Experiment
import tensorflow as tf
import pandas as pd

from DeepTreeAttention.trees import AttentionModel
from DeepTreeAttention.models import vanilla
from DeepTreeAttention.callbacks import callbacks

sleep(randint(0,20))
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = "{}/{}".format("/orange/idtrees-collab/DeepTreeAttention/snapshots/",timestamp)
os.mkdir(save_dir)

experiment = Experiment(project_name="neontrees", workspace="bw4sz")
experiment.add_tag("Train")

#Create output folder
experiment.log_parameter("timestamp",timestamp)
experiment.log_parameter("log_dir",save_dir)

#Create a class and run
model = AttentionModel(config="/home/b.weinstein/DeepTreeAttention/conf/tree_config.yml", log_dir=save_dir)
model.read_data("HSI")
model.create()

baseline = vanilla.create(height=model.config["train"]["HSI"]["crop_size"],width=model.config["train"]["HSI"]["crop_size"],channels=model.config["train"]["HSI"]["sensor_channels"], classes=model.classes)
baseline.compile(
    loss="categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(
    lr=float(model.config["train"]["learning_rate"])),
    metrics=[tf.keras.metrics.CategoricalAccuracy(name='acc')])

labeldf = pd.read_csv(model.classes_file)
label_names = list(labeldf.taxonID.values)

callback_list = callbacks.create(
    experiment = experiment,
    train_data = model.train_split,
    validation_data = model.val_split,
    train_shp = model.train_shp,
    log_dir=None,
    label_names=label_names,
    submodel=False)

baseline.fit(
    model.train_split,
    epochs=model.config["train"]["ensemble"]["epochs"],
    validation_data=model.val_split,
    callbacks=callback_list)

    