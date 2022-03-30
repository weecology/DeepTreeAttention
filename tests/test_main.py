import geopandas as gpd
import os
import pandas as pd
from pytorch_lightning import Trainer

def test_fit(config, m, dm, comet_logger):
    trainer = Trainer(fast_dev_run=True, logger=comet_logger)
    trainer.fit(m,datamodule=dm)
    
def test_predict_dataloader(config, m, dm, comet_logger, ROOT):
    if comet_logger:
        experiment = comet_logger.experiment
    else:
        experiment = None
    df = m.predict_dataloader(
        dm.val_dataloader())
    input_data = pd.read_csv("{}/tests/data/processed/test.csv".format(ROOT))    
    
    assert df.shape[0] == len(input_data.image_path.apply(lambda x: os.path.basename(x).split("_")[0]).unique())
    
def test_evaluate_crowns(config, comet_logger, m, dm, ROOT):
    if comet_logger:
        experiment = comet_logger.experiment
    else:
        experiment = None    
    m.ROOT = "{}/tests".format(ROOT)
    df = m.evaluate_crowns(data_loader = dm.val_dataloader(), crowns=dm.crowns, experiment=experiment)
    assert all(["top{}_score".format(x) in df.columns for x in [1,2]]) 

def test_predict_xy(config, m, dm, ROOT):
    csv_file = "{}/tests/data/sample_neon.csv".format(ROOT)            
    df = pd.read_csv(csv_file)
    label, score = m.predict_xy(coordinates=(df.itcEasting[0],df.itcNorthing[0]))
    
    assert label in dm.species_label_dict.keys()
    assert score > 0 

def test_predict_crown(config, m, dm, ROOT):
    gdf = gpd.read_file("{}/tests/data/crown.shp".format(ROOT))
    label, score = m.predict_crown(geom = gdf.geometry[0], sensor_path = "{}/tests/data/2019_D01_HARV_DP3_726000_4699000_image_crop.tif".format(ROOT))
    
    assert label in dm.species_label_dict.keys()
    assert score > 0 
    
