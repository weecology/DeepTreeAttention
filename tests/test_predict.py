#test_predict
from src import predict
from src import data
from src.models import Hang2020
from src.main import TreeModel
from pytorch_lightning import Trainer
import os
import tempfile
import pytest

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
def model_path(config, dm):
    model = Hang2020.vanilla_CNN(bands=3, classes=3)
    m = TreeModel(model=model, classes=3, config=config, label_dict=dm.species_label_dict)
    m.ROOT = "{}/tests/".format(ROOT)
    filepath = "{}/model.pl".format(tempfile.gettempdir())
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(m, dm)
    trainer.save_checkpoint(filepath)
    
    return filepath

def test_predict_tile(model_path, config):
    PATH =  "{}/tests/data/2019_D01_HARV_DP3_726000_4699000_image_crop.tif".format(ROOT)
    trees = predict.predict_tile(PATH, model_path=model_path, config=config)
    assert all([x in trees.columns for x in ["pred_taxa_top1","geometry","top1_score"]])