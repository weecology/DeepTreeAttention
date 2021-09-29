#Train
import comet_ml
from src import main
from src import data
from src import start_cluster
from src.models import metadata
import torch
from pytorch_lightning import Trainer
import os
import subprocess
from pytorch_lightning.loggers import CometLogger
import pandas as pd
from pandas.util import hash_pandas_object
import numpy as np

#Create datamodule
client = start_cluster.start(cpus=250, mem_size="5GB")
#client = None
data_module = data.TreeData(csv_file="data/raw/neon_vst_data_2021.csv", regenerate=True, client=client, metadata=True)
data_module.setup()
comet_logger = CometLogger(project_name="DeepTreeAttention", workspace=data_module.config["comet_workspace"],auto_output_logging = "simple")
if client:
    client.close()

comet_logger.experiment.log_parameter("commit hash",subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip())

#Hash train and test
train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")
comet_logger.experiment.log_parameter("train_hash",hash_pandas_object(train))
comet_logger.experiment.log_parameter("test_hash",hash_pandas_object(test))
comet_logger.experiment.log_table("train.csv", train)
comet_logger.experiment.log_table("test.csv", test)

model = metadata.metadata_sensor_fusion(sites=data_module.num_sites, classes=data_module.num_classes, bands=data_module.config["bands"])
m = metadata.MetadataModel(
    model=model, 
    classes=data_module.num_classes, 
    label_dict=data_module.species_label_dict, 
    config=data_module.config)

comet_logger.experiment.log_parameters(m.config)

#Create trainer
trainer = Trainer(
    gpus=data_module.config["gpus"],
    fast_dev_run=data_module.config["fast_dev_run"],
    max_epochs=data_module.config["epochs"],
    accelerator=data_module.config["accelerator"],
    checkpoint_callback=False,
    logger=comet_logger)

trainer.fit(m, datamodule=data_module)
results = m.evaluate_crowns(data_module.val_dataloader(), experiment=comet_logger.experiment)

predictions = np.concatenate(predictions)
predictions = np.argmax(predictions, 1)

#Confusion matrix
comet_logger.experiment.log_confusion_matrix(
    test.label.values,
    predictions,
    labels=list(data_module.species_label_dict.keys()),
    max_categories=len(data_module.species_label_dict.keys())
)  

#Log spectral spatial weight
alpha_weight = m.model.weighted_average.detach().numpy()
comet_logger.experiment.log_parameter("spectral_spatial weight", alpha_weight)

#Log prediction
comet_logger.experiment.log_table("test_predictions.csv", results)
