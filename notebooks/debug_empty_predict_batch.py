import geopandas
import tempfile
import tarfile
from src import predict
from src.models import multi_stage
from src.data import read_config
from pytorch_lightning import Trainer
import os

trainer = Trainer()
config = read_config("config.yml")
model_path = "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/ce50bb593a484a28b346e7efe357e0fa_BLAN.pt"
basename = os.path.splitext(os.path.basename(model_path))[0]
site="BLAN"
config["crop_dir"] = os.path.join(config["data_dir"], "5de61342dca34bf894955e4da1e88311")
config["head_class_minimum_ratio"] = 0.25
m = multi_stage.MultiStage.load_from_checkpoint(model_path, config=config)
crown_annotations_path = "/blue/ewhite/b.weinstein/DeepTreeAttention/results/site_crops/BLAN/shp/2022_BLAN_5_754000_4324000_image.shp"

species_prediction = predict.predict_tile(
    crown_annotations=crown_annotations_path,
    filter_dead=True,
    trainer=trainer,
    m=m,
    savedir=tempfile.gettempdir(),
    site="BLAN",
    config=config)

