import comet_ml
import geopandas as gpd
from src import main
from src.models import Hang2020
from src import data
import os
import pytest
import pandas as pd
from pytorch_lightning import Trainer
import tempfile

ROOT = os.path.dirname(os.path.dirname(data.__file__))
os.environ['KMP_DUPLICATE_LIB_OK']='True'

@pytest.fixture(scope="module")
def config():
    #Turn of CHM filtering for the moment
    config = data.read_config(config_path="{}/config.yml".format(ROOT))
    config["min_CHM_height"] = None
    config["iterations"] = 1
    config["rgb_sensor_pool"] = "{}/tests/data/*.tif".format(ROOT)
    config["HSI_sensor_pool"] = "{}/tests/data/*.tif".format(ROOT)
    config["min_samples"] = 1
    config["crop_dir"] = tempfile.gettempdir()
    config["bands"] = 3
    config["classes"] = 3
    config["top_k"] = 1
    config["convert_h5"] = False
    config["plot_n_individuals"] = 1
    
    return config

#Data module
@pytest.fixture(scope="module")
def dm(config):
    csv_file = "{}/tests/data/sample_neon.csv".format(ROOT)           
    if not "GITHUB_ACTIONS" in os.environ:
        regen = False
    else:
        regen = True
    
    dm = data.TreeData(config=config, csv_file=csv_file, regenerate=regen, data_dir="{}/tests/data".format(ROOT)) 
    dm.setup()    
    
    return dm


@pytest.fixture()
def experiment():
    if not "GITHUB_ACTIONS" in os.environ:
        from pytorch_lightning.loggers import CometLogger        
        COMET_KEY = os.getenv("COMET_KEY")
        comet_logger = CometLogger(api_key=COMET_KEY,
                                   project_name="DeepTreeAttention", workspace="bw4sz",auto_output_logging = "simple")
        return comet_logger.experiment
    else:
        return None

#Training module
@pytest.fixture(scope="module")
def m(config, dm):
    model = Hang2020.vanilla_CNN(bands=3, classes=3)
    m = main.TreeModel(model=model, classes=3, config=config, label_dict=dm.species_label_dict)
    m.ROOT = "{}/tests/".format(ROOT)
    
    return m

def test_fit(config, m, dm):
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(m,datamodule=dm)
    
def test_predict_dataloader(config, m, dm, experiment):
    df = m.predict_dataloader(dm.val_dataloader(), experiment = experiment)
    input_data = pd.read_csv("{}/tests/data/processed/test.csv".format(ROOT))    
    
    assert df.shape[0] == len(input_data.image_path.apply(lambda x: os.path.basename(x).split("_")[0]).unique())
    
def test_evaluate_crowns(config, experiment, m, dm):
    m.ROOT = "{}/tests".format(ROOT)
    df = m.evaluate_crowns(data_loader = dm.val_dataloader(), experiment=experiment)
    assert all(["top{}_score".format(x) in df.columns for x in [1,2]]) 

def test_predict_xy(config, m, dm):
    csv_file = "{}/tests/data/sample_neon.csv".format(ROOT)            
    df = pd.read_csv(csv_file)
    label, score = m.predict_xy(coordinates=(df.itcEasting[0],df.itcNorthing[0]))
    
    assert label in dm.species_label_dict.keys()
    assert score > 0 

def test_predict_crown(config, m, dm):
    gdf = gpd.read_file("{}/tests/data/crown.shp".format(ROOT))
    label, score = m.predict_crown(geom = gdf.geometry[0], sensor_path = "{}/tests/data/2019_D01_HARV_DP3_726000_4699000_image_crop.tif".format(ROOT))
    
    assert label in dm.species_label_dict.keys()
    assert score > 0 
    
