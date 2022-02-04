#Train
import comet_ml
import glob
import geopandas as gpd
from src import main
from src import data
from src import start_cluster
from src.models import Hang2020
from src import visualize
from src import metrics
from pytorch_lightning import Trainer
import subprocess
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks import LearningRateMonitor
import pandas as pd
from pandas.util import hash_pandas_object

#Create datamodule
config = data.read_config("config.yml")
if config["regenerate"]:
    client = start_cluster.start(cpus=75, mem_size="5GB")
else:
    client = None

comet_logger = CometLogger(project_name="DeepTreeAttention", workspace=config["comet_workspace"],auto_output_logging = "simple")    
data_module = data.TreeData(csv_file="data/raw/neon_vst_data_2022.csv", client=client, metadata=True, comet_logger=comet_logger)
data_module.setup()
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
novel = pd.read_csv("data/processed/novel_species.csv")

comet_logger.experiment.log_parameter("train_hash",hash_pandas_object(train))
comet_logger.experiment.log_parameter("test_hash",hash_pandas_object(test))
comet_logger.experiment.log_parameter("num_species",data_module.num_classes)
comet_logger.experiment.log_table("train.csv", train)
comet_logger.experiment.log_table("test.csv", test)
comet_logger.experiment.log_table("novel_species.csv", novel)

#model = metadata.metadata_sensor_fusion(sites=data_module.num_sites, classes=data_module.num_classes, bands=data_module.config["bands"])
model = Hang2020.Hang2020(classes=data_module.num_classes, bands=data_module.config["bands"])
m = main.TreeModel(
    model=model, 
    classes=data_module.num_classes, 
    label_dict=data_module.species_label_dict)

comet_logger.experiment.log_parameters(m.config)

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
alpha_weight = m.model.weighted_average.detach().numpy()
comet_logger.experiment.log_parameter("spectral_spatial weight", alpha_weight)

#Log prediction
comet_logger.experiment.log_table("test_predictions.csv", results)

#Within site confusion
site_lists = train.groupby("label").site.unique()
within_site_confusion = metrics.site_confusion(y_true = results.label, y_pred = results.pred_label_top1, site_lists=site_lists)
comet_logger.experiment.log_metric("within_site_confusion", within_site_confusion)

#Within plot confusion
plot_lists = train.groupby("label").plotID.unique()
within_plot_confusion = metrics.site_confusion(y_true = results.label, y_pred = results.pred_label_top1, site_lists=plot_lists)
comet_logger.experiment.log_metric("within_plot_confusion", within_plot_confusion)

