#test_predict
from src import predict
from src import data
from src.models import Hang2020
from src.models import dead
from src.main import TreeModel
from src.predict import on_the_fly_dataset
import rasterio as rio
import torch
from pytorch_lightning import Trainer
import numpy as np
import os
import tempfile
import pytest
import pandas as pd
import geopandas as gpd
from skimage import io

ROOT = os.path.dirname(os.path.dirname(data.__file__))        

@pytest.fixture()
def config():
    #Turn of CHM filtering for the moment
    config = data.read_config(config_path="{}/config.yml".format(ROOT))
    config["min_CHM_height"] = None
    config["iterations"] = 1
    config["rgb_sensor_pool"] = "{}/tests/data/*.tif".format(ROOT)
    config["HSI_sensor_pool"] = "{}/tests/data/*.tif".format(ROOT)
    config["min_train_samples"] = 1
    config["min_test_samples"] = 1    
    config["crop_dir"] = tempfile.gettempdir()
    config["bands"] = 3
    config["classes"] = 2
    config["top_k"] = 1
    config["convert_h5"] = False
    config["plot_n_individuals"] = 1
    config["gpus"] = 0
    config["include_outliers"] = True
    config["megaplot_dir"] = None
    config["workers"] = 0
    
    return config

#Data module
@pytest.fixture()
def dm(config):
    csv_file = "{}/tests/data/sample_neon.csv".format(ROOT)           
    if not "GITHUB_ACTIONS" in os.environ:
        regen = True
    else:
        regen = True
    
    dm = data.TreeData(config=config, csv_file=csv_file, regenerate=regen, data_dir="{}/tests/data".format(ROOT), debug=True) 
    dm.setup()    
    
    return dm

#Training module
@pytest.fixture()
def species_model_path(config, dm):
    model = Hang2020.vanilla_CNN(bands=3, classes=3)
    m = TreeModel(model=model, classes=3, config=config, label_dict=dm.species_label_dict)
    m.ROOT = "{}/tests/".format(ROOT)
    filepath = "{}/model.pl".format(tempfile.gettempdir())
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(m, dm)
    trainer.save_checkpoint(filepath)
    
    return filepath

#Training module
@pytest.fixture()
def dead_model_path():
    m = dead.AliveDead()
    shp = gpd.read_file("{}/tests/data/processed/crowns.shp".format(ROOT))
    shp["label"] = "Alive"
    shp.loc[0,"label"] = "Dead"
    shp["image_path"] = "{}/tests/data/2019_HARV_6_725000_4700000_image.tif".format(ROOT)
    tile_bounds = rio.open("{}/tests/data/2019_HARV_6_725000_4700000_image.tif".format(ROOT)).bounds 
    coords = shp.geometry.bounds
    coords = coords.rename(columns = {"minx":"xmin","miny":"ymin","maxx":"xmax","maxy":"ymax"})
    coords["xmin"] = (coords["xmin"] - tile_bounds.left) * 10
    coords["xmax"] = (coords["xmax"] - tile_bounds.left) * 10
    coords["ymin"] = (tile_bounds.top - coords["ymax"]) * 10
    coords["ymax"] = (tile_bounds.top - coords["ymin"] ) * 10
    
    csv = pd.concat([shp[["image_path","label"]],coords], axis=1)
    csv_file = "{}/dead.csv".format(tempfile.gettempdir())
    csv.to_csv(csv_file)
    ds = dead.AliveDeadDataset(csv_file, root_dir=ROOT)
    train_loader = torch.utils.data.DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )     
    
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(m, train_loader, train_loader)    
    filepath = "{}/dead_model.pl".format(tempfile.gettempdir())
    trainer.save_checkpoint(filepath)
    
    return filepath

def test_dead_tree_model(dead_model_path):
    dead_model_path = "/Users/benweinstein/Downloads/f4f3664646684a4d9eeff616415960a2.pl"
    m = dead.AliveDead.load_from_checkpoint(dead_model_path)
    m.eval()
    dead_tree = io.imread("{}/tests/data/dead_tree.png".format(ROOT))
    transform = dead.get_transform(augment=False)
    dead_tree_transformed = transform(dead_tree)
    score = m(dead_tree_transformed.unsqueeze(0)).detach()
    assert np.argmax(score) == 1
    
    dead_tree = io.imread("{}/tests/data/dead_tree2.png".format(ROOT))
    transform = dead.get_transform(augment=False)
    dead_tree_transformed = transform(dead_tree)
    score = m(dead_tree_transformed.unsqueeze(0)).detach()
    assert np.argmax(score) == 1
    
    alive_tree = io.imread("{}/tests/data/alive_tree.png".format(ROOT))
    alive_tree = transform(alive_tree)
    score = m(alive_tree.unsqueeze(0)).detach()
    assert np.argmax(score) == 0
    
def test_predict_tile(species_model_path, dead_model_path, config):
    PATH =  "{}/tests/data/2019_D01_HARV_DP3_726000_4699000_image_crop.tif".format(ROOT)
    #HOTFIX
    dead_model_path = "/Users/benweinstein/Downloads/f4f3664646684a4d9eeff616415960a2.pl"
    trees = predict.predict_tile(PATH, dead_model_path = dead_model_path, species_model_path=species_model_path, config=config)
    assert all([x in trees.columns for x in ["pred_taxa_top1","geometry","top1_score","dead_label"]])