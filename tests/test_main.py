#test main module
from src import main
from src.models import Hang2020
from src import data
import os
import pytest

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
    

def test_TreeModel(config):
    m = main.TreeModel(model=Hang2020.vanilla_CNN, config=config)
    