#Test semi_supervised.py
from pytorch_lightning import Trainer
from src import semi_supervised
from src.models import Hang2020, baseline
import pytest

@pytest.fixture()
def m(dm, config, ROOT):    
    model = Hang2020.spectral_network(bands=config["bands"], classes=dm.num_classes)    
    m = baseline.TreeModel(
        model=model, 
        config=config,
        classes=dm.num_classes, 
        loss_weight=None,
        label_dict=dm.species_label_dict)
        
    return m

def test_create_dataframe(config, m, dm):
    config["semi_supervised"]["crop_dir"] = config["crop_dir"]
    dm.train = semi_supervised.create_dataframe(config, m=m, unlabeled_df=dm.train, label_to_taxon_id=dm.label_to_taxonID)
    dm.train.shape[0] == config["semi_supervised"]["num_samples"]
