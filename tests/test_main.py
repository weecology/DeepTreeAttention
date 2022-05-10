import geopandas as gpd
import os
import pandas as pd
from pytorch_lightning import Trainer
from src import utils

def test_fit(config, m, dm):
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(m,datamodule=dm)
    
def test_predict_dataloader(config, m, dm, experiment, ROOT):
    df = m.predict_dataloader(dm.val_dataloader(), test_crowns=dm.crowns, test_points=dm.canopy_points, experiment = experiment)
    input_data = pd.read_csv("{}/tests/data/processed/test.csv".format(ROOT))    
    
    assert df.shape[0] == dm.test.shape[0]
    
def test_evaluate_crowns(config, experiment, m, dm, ROOT):
    m.ROOT = "{}/tests".format(ROOT)
    df = m.evaluate_crowns(data_loader = dm.val_dataloader(), crowns=dm.crowns, points=dm.canopy_points, experiment=experiment)
    assert all(["top{}_score".format(x) in df.columns for x in [1,2]]) 
    
    
