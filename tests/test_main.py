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
    
    assert df.shape[0] == len(input_data.image_path.apply(lambda x: os.path.basename(x).split("_")[0]))
    
def test_evaluate_crowns(config, experiment, m, dm, ROOT):
    m.ROOT = "{}/tests".format(ROOT)
    df = m.evaluate_crowns(data_loader = dm.val_dataloader(), crowns=dm.crowns, points=dm.canopy_points, experiment=experiment)
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

def test_ensemble(m, dm, ROOT):
    m.ROOT = "{}/tests".format(ROOT)
    models = [m, m]
    
    individuals = {}
    result_df = []
    for x in models:
        results, features = x.predict_dataloader(
            data_loader=dm.val_dataloader(),
            experiment=None,
            return_features=True
        )
        
        for index, row in enumerate(features):
            try:
                individuals[results.individual.iloc[index]].append(row)
            except:
                individuals[results.individual.iloc[index]] = [row]
                
        results = x.evaluate_crowns(
            dm.val_dataloader(),
            crowns=dm.crowns
        )
        result_df.append(results)
        
    result_df = pd.concat(result_df)
    temporal_df = utils.ensemble(result_df, individuals)
    assert temporal_df.shape[0] == dm.test.shape[0] * len(models)
    
    
