#Test metadata model
from src.models import metadata
from src import data
import torch
import tempfile
import os
import pytest

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
    
    dm = data.TreeData(config=config, csv_file=csv_file, regenerate=regen, data_dir="{}/tests/data".format(ROOT)) 
    dm.setup()    
    
    return dm

def test_metadata():
    m = metadata.metadata(sites=23, classes=10)
    sites = torch.zeros(20, 23)
    for x in range(sites.shape[0]):
        sites[x,torch.randint(low=0,high=23,size=(1,))] =1         
    output = m(sites)
    assert output.shape == (20,10)
    
def test_metadata_sensor_fusion():
    sites = torch.zeros(20, 23)
    image = torch.randn(20, 3, 11, 11)    
    
    m = metadata.metadata_sensor_fusion(sites=23, bands=3, classes=10)
    prediction = m(image, sites)
    assert prediction.shape == (20,10)

def test_MetadataModel(dm):
    model = metadata.metadata_sensor_fusion(sites=1, bands=3, classes=2)
    m = metadata.MetadataModel(model=model, sites=1, classes=2, label_dict=dm.species_label_dict)
    m.fit(data_module = m)