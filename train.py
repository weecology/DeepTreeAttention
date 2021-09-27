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
COMET_KEY = os.getenv("COMET_KEY")
#client = start_cluster.start(cpus=250, mem_size="5GB")
client = None
data_module = data.TreeData(csv_file="data/raw/neon_vst_data_2021.csv", regenerate=False, client=client, metadata=True)
data_module.setup()
comet_logger = CometLogger(api_key=COMET_KEY,
                           project_name="DeepTreeAttention", workspace=data_module.config["comet_workspace"],auto_output_logging = "simple")
if client:
    client.close()

resampled_data = data_module.resample(csv_file="data/processed/train.csv", oversample=True)
resampled_data.to_csv("data/processed/resampled_train.csv", index=False)

comet_logger.experiment.log_parameter("commit hash",subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip())

#Override train file with resampling
data_module.train_file = "data/processed/resampled_train.csv"

#Hash train and test
train = pd.read_csv("data/processed/resampled_train.csv")
test = pd.read_csv("data/processed/test.csv")
comet_logger.experiment.log_parameter("train_hash",hash_pandas_object(train))
comet_logger.experiment.log_parameter("test_hash",hash_pandas_object(test))
comet_logger.experiment.log_table("resampled_train.csv", train)
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
results, crown_metrics = m.evaluate_crowns("data/processed/test.csv", experiment=comet_logger.experiment)
comet_logger.experiment.log_metrics(crown_metrics)

m.eval()
predictions = []
for batch in data_module.val_dataloader():
    inputs, targets = batch
    site = inputs["site"]
    images = inputs["HSI"]
    with torch.no_grad():
        pred = m.model(images, site)
    predictions.append(pred)

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
