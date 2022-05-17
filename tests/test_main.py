import geopandas as gpd
import os
import pandas as pd
from pytorch_lightning import Trainer
from src import utils

def test_fit(config, m, dm):
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(m,datamodule=dm)