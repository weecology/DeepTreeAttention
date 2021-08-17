#test main module
from src import main
from src.models import Hang2020
from src import data
import os
import pytest
from pytorch_lightning import Trainer

ROOT = os.path.dirname(os.path.dirname(data.__file__))

@pytest.fixture()
def config(tmpdir):
    #Turn of CHM filtering for the moment
    config = data.read_config(config_path="{}/config.yml".format(ROOT))
    config["min_CHM_height"] = None
    config["iterations"] = 1
    config["rgb_sensor_pool"] = "{}/tests/data/*.tif".format(ROOT)
    config["HSI_sensor_pool"] = "{}/tests/data/*.tif".format(ROOT)
    config["min_samples"] = 1
    config["crop_dir"] = tmpdir
    config["bands"] = 3
    config["classes"] = 2
    
    return config
    
def test_fit(config):
    csv_file = "{}/tests/data/sample_neon.csv".format(ROOT)    
    m = main.TreeModel(model=Hang2020.vanilla_CNN, config=config)
    dm = data.TreeData(config=config, csv_file=csv_file, regenerate=True, data_dir="{}/tests/data".format(ROOT))
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(m,datamodule=dm)
    