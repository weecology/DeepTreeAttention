#Train
import comet_ml
import glob
import geopandas as gpd
from src import main
from src import data
from src import start_cluster
from src.models import metadata
from src import visualize
from src import metrics
from pytorch_lightning import Trainer
import subprocess
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.profiler import AdvancedProfiler

import pandas as pd
from pandas.util import hash_pandas_object

#Create datamodule
config = data.read_config("config.yml")
if config["regenerate"]:
    client = start_cluster.start(cpus=50, mem_size="5GB")
else:
    client = None
data_module = data.TreeData(csv_file="data/raw/neon_vst_data_2021.csv", regenerate=config["regenerate"], client=client, metadata=True)
data_module.setup()
comet_logger = CometLogger(project_name="DeepTreeAttention", workspace=data_module.config["comet_workspace"],auto_output_logging = "simple")
if client:
    client.close()

comet_logger.experiment.log_parameter("commit hash",subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip())

#log branch
git_branch = subprocess.check_output(["git","symbolic-ref", "--short", "HEAD"]).decode("utf8")[0:-1]
comet_logger.experiment.log_parameter("git branch",git_branch)
comet_logger.experiment.add_tag(git_branch)

#Hash train and test
train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")
novel = gpd.read_file("data/processed/novel_species.shp")
novel = pd.DataFrame(novel)

comet_logger.experiment.log_parameter("train_hash",hash_pandas_object(train))
comet_logger.experiment.log_parameter("test_hash",hash_pandas_object(test))
comet_logger.experiment.log_parameter("num_species",data_module.num_classes)
comet_logger.experiment.log_table("train.csv", train)
comet_logger.experiment.log_table("test.csv", test)
comet_logger.experiment.log_table("novel_species.csv", novel)

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

rgb_pool = glob.glob(data_module.config["rgb_sensor_pool"], recursive=True)

visualize.confusion_matrix(
    comet_experiment=comet_logger.experiment,
    results=results,
    species_label_dict=data_module.species_label_dict,
    test_crowns="data/processed/crowns.shp",
    test_csv="data/processed/test.csv",
    test_points="data/processed/canopy_points.shp",
    rgb_pool=rgb_pool
)

#Log spectral spatial weight
alpha_weight = m.model.sensor_model.weighted_average.detach().numpy()
comet_logger.experiment.log_parameter("spectral_spatial weight", alpha_weight)

#Log prediction
comet_logger.experiment.log_table("test_predictions.csv", results)

#Within site confusion
site_lists = train.groupby("label").site.unique()
within_site_confusion = metrics.site_confusion(y_true = results.label, y_pred = results.pred_label, site_lists=site_lists)
comet_logger.experiment.log_metric("within_site_confusion", within_site_confusion)