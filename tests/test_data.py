#Test data module
import pandas as pd
from src import data

def test_TreeData_setup(dm, ROOT):    
    dm.setup("fit")
    test = dm.test
    train = dm.train
    
    assert not test.empty
    assert not train.empty
    assert not any([x in train.image_path.unique() for x in test.image_path.unique()])
    
def test_TreeDataset(m, dm, config):
    #Train loader
    ds = data.TreeDataset(df=dm.train, config=dm.config)
    individuals, inputs, label = ds[0]    
    assert inputs["HSI"].shape == (config["bands"], config["image_size"], config["image_size"])
    
def test_sample_plots(dm, config):
    train, test = data.sample_plots(shp=dm.crowns, min_test_samples=2, min_train_samples=2)
    assert not train.empty
    assert train[train.individual.isin(test.individual)].empty
