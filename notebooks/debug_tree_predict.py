# Debug TREE predict
from src.predict import generate_prediction_crops
from src.data import read_config
from src.utils import create_glob_lists
from src import predict
from src.models import multi_stage
from pytorch_lightning import Trainer
import os
import tempfile

config = read_config("config.yml")
rgb_pool, h5_pool, hsi_pool, CHM_pool = create_glob_lists(config)
crop_dir = "/blue/ewhite/b.weinstein/DeepTreeAttention/results/site_crops/TREE/2022_STEI_5_306000_5036000_image"
os.environ["TMPDIR"] = tempfile.gettempdir()


crown_annotations_path = generate_prediction_crops(
    crown_path="/blue/ewhite/b.weinstein/DeepTreeAttention/results/crowns/2022_STEI_5_306000_5036000_image.shp",
    config=config,
    crop_dir=crop_dir,
    rgb_pool=rgb_pool,
    h5_pool=h5_pool,
    img_pool=hsi_pool,
    as_numpy=True
)

model_path = "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/990e6f1101b2423f86d4cd16f373deab_TREE.pt"
m = multi_stage.MultiStage.load_from_checkpoint(model_path, config=config)
trainer = Trainer(devices=config["gpus"])

species_prediction = predict.predict_tile(
    crown_annotations=crown_annotations_path,
    filter_dead=True,
    trainer=trainer,
    m=m,
    savedir=tempfile.gettempdir(),
    site="TREE",
    config=config)