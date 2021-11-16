#Simulation 
import os
from src import simulation, data

ROOT = os.path.dirname(os.path.dirname(simulation.__file__))
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def test_run():
    config = data.read_config("{}/simulation.yml".format(ROOT))
    config["autoencoder_epochs"] = 1
    config["classifier_epochs"] = 1    
    config["workers"] = 0
    config["gpus"] = 0
    config["fast_dev_run"] = False
    config["proportion_switch"] = 0.1
    config["samples"] = 100
    s = simulation.run(ID="test", config=config)
    
    