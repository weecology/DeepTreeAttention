#Train
import comet_ml
from src import main
from src import data
from src import start_cluster
from src.models import Hang2020
from pytorch_lightning import Trainer
import os
import subprocess
from pytorch_lightning.loggers import CometLogger
import pandas as pd
from pandas.util import hash_pandas_object

COMET_KEY = os.getenv("COMET_KEY")
comet_logger = CometLogger(api_key=COMET_KEY,
                            project_name="DeepTreeAttention", workspace="bw4sz",auto_output_logging = "simple")
comet_logger.experiment.log_parameter("commit hash",subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip())

#Create datamodule
client = start_cluster.start(cpus=80)
data_module = data.TreeData(csv_file="data/raw/neon_vst_data_2021.csv", regenerate=True, client=client)
data_module.setup()

#Hash train and test
train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")
comet_logger.experiment.log_parameter("train_hash",hash_pandas_object(train))
comet_logger.experiment.log_parameter("test_hash",hash_pandas_object(test))

m = main.TreeModel(model=Hang2020.vanilla_CNN, bands=data_module.config["bands"], classes=data_module.num_classes,label_dict=data_module.species_label_dict)
comet_logger.experiment.log_parameters(m.config)

#Create trainer
trainer = Trainer(
    gpus=data_module.config["gpus"],
    fast_dev_run=data_module.config["fast_dev_run"],
    max_epochs=data_module.config["epochs"],
    accelerator=data_module.config["accelerator"],
    logger=comet_logger)

trainer.fit(m, datamodule=data_module)
results, crown_metrics = m.evaluate_crowns("data/processed/test.csv")
comet_logger.experiment.log_metrics(crown_metrics)

#Confusion matrix
comet_logger.experiment.log_confusion_matrix(
    results.true_label,
    results.label,
    labels=list(data_module.species_label_dict.keys()))  
