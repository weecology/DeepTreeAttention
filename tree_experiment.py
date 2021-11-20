#Train
import comet_ml
import glob
from src import data
from src import start_cluster
from src.models import metadata
from src import visualize
from src import metrics

from pytorch_lightning import Trainer
import subprocess
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks import LearningRateMonitor
import pandas as pd
from pandas.util import hash_pandas_object
from datetime import datetime

def run():
    """Run a tree experiment"""
    #Create datamodule
    client = start_cluster.start(cpus=75, mem_size="5GB")
    #client = None
    
    now = datetime.now() # current date and time
    timestamp = now.strftime("%H:%M:%S")
    config = data.read_config("config.yml")
    comet_logger = CometLogger(project_name="DeepTreeAttention", workspace=config["comet_workspace"],auto_output_logging = "simple")
    comet_logger.experiment.add_tag("Autoencoder")
    data_module = data.TreeData(csv_file="data/raw/neon_vst_data_2021.csv", regenerate=True, client=client, metadata=True, comet_logger=comet_logger)
    data_module.setup()
        
    if client:
        client.close()
        
    rows = []
    for x in [False, True]:
        for x in range(2):
            comet_logger = CometLogger(project_name="DeepTreeAttention", workspace=config["comet_workspace"],auto_output_logging = "simple")        
            comet_logger.experiment.add_tag("Tree Experiment")        
            data_module.config["include_outliers"] = x
            comet_logger.experiment.log_parameter("commit hash",subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip())
            
            #log branch
            git_branch = subprocess.check_output(["git","symbolic-ref", "--short", "HEAD"]).decode("utf8")[0:-1]
            comet_logger.experiment.log_parameter("git branch",git_branch)
            comet_logger.experiment.add_tag(git_branch)
            
            #Hash train and test
            test = pd.read_csv("data/processed/test.csv")
            train = pd.read_csv("data/processed/train.csv")
            novel = pd.read_csv("data/processed/novel_species.csv")
            
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
            
            comet_logger.experiment.log_parameters(data_module.config)
            
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
            
            ##Novel species prediction, get scores
            #novel_prediction = metrics.novel_prediction(model=m.model.sensor_model, csv_file="data/processed/novel_species.csv", config=config)
            #comet_logger.experiment.log_table("novel_prediction.csv", novel_prediction)
            #mean_novel_prediction = novel_prediction.softmax_score.mean()
            #comet_logger.experiment.log_metric(name="Mean unknown species softmax score", value=mean_novel_prediction)
            
            row = pd.DataFrame({"timestamp":[timestamp],"Outliers": [x], "Micro Accuracy":m.metrics["Micro Accuracy"],"Macro Accuracy":m.metrics["Macro Accuracy"]})
            rows.append(row)
    
    rows = pd.concat(rows)
    rows.to_csv("results/experiment_{}.csv".format(comet_logger.experiment.get_key()))
    
for i in range(10):
    run()
