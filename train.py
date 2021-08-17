#Train
import comet_ml
from src import main
from src import data
from pytorch_lightning import Trainer
import os
import subprocess
from pytorch_lightning.loggers import CometLogger

COMET_KEY = os.getenv("COMET_KEY")
comet_logger = CometLogger(api_key=COMET_KEY,
                            project_name="DeepTreeAttention", workspace="bw4sz",auto_output_logging = "simple")
comet_logger.experiment.log_parameter("commit hash",subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip())

#Create datamodule
data_module = data.TreeData()

#Create model
m = main.TreeModel()
comet_logger.experiment.log_parameters(m.config)

#Create trainer
trainer = Trainer(
    gpus=data_module.config["gpus"],
    fast_dev_run=data_module.config["fast_dev_run"],
    max_epochs=data_module.config["epochs"],
    logger=comet_logger)

trainer.fit(m, datamodule=data_module)
trainer.test(test_dataloaders=data_module.val_dataloader())