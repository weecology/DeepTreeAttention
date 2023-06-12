from src.models import multi_stage
from src.data import read_config, TreeData
from pytorch_lightning import Trainer
import os

config = read_config("config.yml")
crop_dir = os.path.join(config["data_dir"], config["use_data_commit"])
config["crop_dir"] = crop_dir
data_module = TreeData(
    csv_file="data/raw/neon_vst_data_2022.csv",
    data_dir=config["crop_dir"],
    config=config)

train = data_module.train.copy()
test = data_module.test.copy()
crowns = data_module.crowns.copy()

train["individual"] = train["individualID"]
test["individual"] = test["individualID"]

m = multi_stage.MultiStage(train_df=train, test_df=test, crowns=None, config=config)
trainer = Trainer(fast_dev_run=True)
trainer.fit(m)
trainer.save_checkpoint("example.pl")
m2 = multi_stage.MultiStage.load_from_checkpoint("example.pl")

