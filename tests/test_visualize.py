#test validate
import comet_ml
from src import visualize
from src import data
from src import main
from src.models import Hang2020
import os
import glob
import pytest
import tempfile
ROOT = os.path.dirname(os.path.dirname(visualize.__file__))

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
    config["classes"] = 5
    config["top_k"] = 1
    config["convert_h5"] = False
    
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

#Training module
@pytest.fixture(scope="module")
def m(config, dm):
    model = Hang2020.vanilla_CNN(bands=3, classes=3)
    m = main.TreeModel(model=model, classes=3, config=config, label_dict=dm.species_label_dict)
    m.ROOT = "{}/tests/".format(ROOT)
    
    return m

def test_confusion_matrix(dm, m, experiment):
    if experiment:
        rgb_pool = glob.glob(m.config["rgb_sensor_pool"])
        m.ROOT = "{}/tests".format(ROOT)
        results = m.evaluate_crowns(data_loader = dm.val_dataloader())
        visualize.confusion_matrix(
            comet_experiment=experiment,
            results=results,
            species_label_dict=dm.species_label_dict,
            test_csv="{}/tests/data/processed/test.csv".format(ROOT),
            test_points="{}/tests/data/processed/test_points.shp".format(ROOT),
            test_crowns="{}/tests/data/processed/test_crowns.shp".format(ROOT),
            rgb_pool=rgb_pool)
    else:
        pass