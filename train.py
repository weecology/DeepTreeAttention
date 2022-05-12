#Train
import comet_ml
import glob
import geopandas as gpd
import os
import numpy as np
from src import main
from src import data
from src import start_cluster
from src.models import year
from src import visualize
from src import metrics
import sys
from pytorch_lightning import Trainer
import subprocess
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks import LearningRateMonitor
import pandas as pd
from pandas.util import hash_pandas_object
import torchmetrics

#Get branch name for the comet tag
git_branch=sys.argv[1]
git_commit=sys.argv[2]

#Create datamodule
config = data.read_config("config.yml")
comet_logger = CometLogger(project_name="DeepTreeAttention", workspace=config["comet_workspace"], auto_output_logging="simple")    

#Generate new data or use previous run
if config["use_data_commit"]:
    config["crop_dir"] = os.path.join(config["data_dir"], config["use_data_commit"])
    client = None    
else:
    crop_dir = os.path.join(config["data_dir"], comet_logger.experiment.get_key())
    os.mkdir(crop_dir)
    client = start_cluster.start(cpus=50, mem_size="4GB")    
    config["crop_dir"] = crop_dir

comet_logger.experiment.log_parameter("git branch",git_branch)
comet_logger.experiment.add_tag(git_branch)
comet_logger.experiment.log_parameter("commit hash",git_commit)
comet_logger.experiment.log_parameters(config)

data_module = data.TreeData(
    csv_file="data/raw/neon_vst_data_2022.csv",
    data_dir=config["crop_dir"],
    config=config,
    client=client,
    metadata=True,
    comet_logger=comet_logger)

data_module.setup()
if client:
    client.close()

comet_logger.experiment.log_parameter("train_hash",hash_pandas_object(data_module.train))
comet_logger.experiment.log_parameter("test_hash",hash_pandas_object(data_module.test))
comet_logger.experiment.log_parameter("num_species",data_module.num_classes)
comet_logger.experiment.log_table("train.csv", data_module.train)
comet_logger.experiment.log_table("test.csv", data_module.test)

if not config["use_data_commit"]:
    comet_logger.experiment.log_table("novel_species.csv", data_module.novel)

#Loss weight, balanced
loss_weight = []
for x in data_module.species_label_dict:
    loss_weight.append(1/data_module.train[data_module.train.taxonID==x].shape[0])
    
loss_weight = np.array(loss_weight/np.max(loss_weight))
#Provide min value
loss_weight[loss_weight < 0.5] = 0.5  

comet_logger.experiment.log_parameter("loss_weight", loss_weight)
m = year.YearEnsemble(classes=data_module.num_classes, years=data_module.train.tile_year.unique(), config=config, label_dict=data_module.species_label_dict)

#Create trainer
lr_monitor = LearningRateMonitor(logging_interval='epoch')
trainer = Trainer(
    gpus=data_module.config["gpus"],
    fast_dev_run=data_module.config["fast_dev_run"],
    max_epochs=data_module.config["epochs"],
    accelerator=data_module.config["accelerator"],
    checkpoint_callback=False,
    callbacks=[lr_monitor],
    logger=comet_logger)

trainer.fit(m, datamodule=data_module)

#Save model checkpoint
trainer.save_checkpoint("/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/{}.pl".format(comet_logger.experiment.id))
predictions = trainer.predict(m, dataloaders=data_module.predict_dataloader(data_module.test))
ensemble_df = m.ensemble(predictions)
ensemble_df["individualID"] = ensemble_df["individual"]
ensemble_df = ensemble_df.merge(data_module.test, on="individualID")
m.ensemble_metrics(ensemble_df, experiment=comet_logger.experiment)

rgb_pool = glob.glob(data_module.config["rgb_sensor_pool"], recursive=True)
visualize.confusion_matrix(
    comet_experiment=comet_logger.experiment,
    results=ensemble_df,
    species_label_dict=data_module.species_label_dict,
    test_crowns=data_module.crowns,
    test=data_module.test,
    test_points=data_module.canopy_points,
    rgb_pool=rgb_pool
)

#Log prediction
comet_logger.experiment.log_table("test_predictions.csv", ensemble_df)

#Cross year prediction
dls = data_module.predict_dataloader(data_module.test)

macro = []
micro = []
train_years = []
test_years = []
for index, year_model in enumerate(m.models):
    train_year = m.years[index]
    yr = main.TreeModel(year_model.model, classes=m.classes, label_dict=data_module.species_label_dict, loss_weight=loss_weight)    
    for test_index, dl in enumerate(dls):
        test_year = m.years[test_index]
        with comet_logger.experiment.context_manager("{}_{}".format(train_year, test_year)):
            results = yr.evaluate_crowns(
                crowns=data_module.crowns,
                experiment = comet_logger.experiment
            )
            micro.append(comet_logger.experiment.get_metric("OSBS_micro"))
            macro.append(comet_logger.experiment.get_metric("OSBS_macro"))
            test_years.append(test_year)
            train_years.append(train_year)

cross_matrix = pd.DataFrame({"train":train_years,"test":test_years,"micro": micro,"macro":macro})
comet_logger.experiment.log_table("cross_matrix.csv", cross_matrix)
    