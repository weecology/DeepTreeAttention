#Test data module
from src import data
import pandas as pd

def test_TreeData_setup(dm, config, ROOT):
    #One site's worth of data
    dm.setup()
    test = pd.read_csv("{}/tests/data/processed/test.csv".format(ROOT))
    train = pd.read_csv("{}/tests/data/processed/train.csv".format(ROOT))
    
    assert not test.empty
    assert not train.empty
    assert not any([x in train.image_path.unique() for x in test.image_path.unique()])
    assert all([x in ["image_path","label","site","taxonID","siteID","plotID","individualID","point_id","box_id"] for x in train.columns])
    
def test_TreeDataset(dm, config,tmpdir, ROOT):
    #Train loader
    data_loader = data.TreeDataset(csv_file="{}/tests/data/processed/train.csv".format(ROOT), config=config, image_size=config["image_size"])
    individuals, inputs, label = data_loader[0]
    image = inputs["HSI"]
    assert image.shape == (3, config["image_size"], config["image_size"])
    
    #Test loader
    data_loader = data.TreeDataset(csv_file="{}/tests/data/processed/test.csv".format(ROOT), train=False, config=config)    
    annotations = pd.read_csv("{}/tests/data/processed/test.csv".format(ROOT))
    
    assert len(data_loader) == annotations.shape[0]
