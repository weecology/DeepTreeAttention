#Test metadata model
from src.models import metadata
from src import data
import torch
import tempfile
import os
import pytest
from pytorch_lightning import Trainer

ROOT = os.path.dirname(os.path.dirname(data.__file__))

@pytest.fixture(scope="session")
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
    config["classes"] = 2
    config["top_k"] = 1
    config["convert_h5"] = False
    
    return config

#Data module
@pytest.fixture(scope="session")
def dm(config):
    csv_file = "{}/tests/data/sample_neon.csv".format(ROOT)           
    if not "GITHUB_ACTIONS" in os.environ:
        regen = False
    else:
        regen = True
    
    dm = data.TreeData(config=config, csv_file=csv_file, regenerate=regen, data_dir="{}/tests/data".format(ROOT), metadata=True) 
    dm.setup()    
    
    return dm

def test_metadata():
    m = metadata.metadata(classes=10)
    sites = torch.zeros(20, 1)     
    output = m(sites)
    assert output.shape == (20,10)
    
def test_metadata_sensor_fusion():
    sites = torch.zeros(20, 1)
    image = torch.randn(20, 3, 11, 11)    
    
    m = metadata.metadata_sensor_fusion(bands=3, classes=10)
    prediction = m(image, sites)
    assert prediction.shape == (20,10)

def test_MetadataModel(config, dm):
    model = metadata.metadata(classes=2)
    m = metadata.MetadataModel(model=model, classes=2, label_dict=dm.species_label_dict, config=config)
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(m,datamodule=dm)    
