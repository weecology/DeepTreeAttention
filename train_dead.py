# Train Dead 
import comet_ml
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CometLogger
from src.models import dead
from src.data import read_config

config = read_config("config.yml")
comet_logger = CometLogger(
    project_name="DeepTreeAttention",
    workspace=config["comet_workspace"],
    auto_output_logging="simple"
)    
comet_logger.experiment.add_tag("Dead")

trainer = Trainer(max_epochs=config["dead"]["epochs"], checkpoint_callback=False, gpus=self.config["gpus"])
m = dead.AliveDead(config=config)

trainer.fit(m)
trainer.validate(m)
trainer.test(m)
trainer.save_checkpoint("{}/{}.pl".format(config["dead"]["savedir"],comet_logger.experiment.id))
