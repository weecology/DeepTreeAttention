#Train
import comet_ml
from src import main, data
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
trainer = m.create_trainer()
trainer.fit(m, datamodule=data_module)
trainer.test(test_dataloaders=data_module.val_dataloader())