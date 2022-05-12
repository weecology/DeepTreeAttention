#test year model
from pytorch_lightning import Trainer
from src.models import year
import torch
import pandas as pd
import numpy as np

def test_YearEnsemble(dm, config):
    years = dm.train.tile_year.unique()      
    m  = year.YearEnsemble(years=years, classes=dm.num_classes, label_dict=dm.species_label_dict, config=config)
    image = torch.randn(20, 3, 11, 11)    
    for x in range(len(years)):
        with torch.no_grad(): 
            output = m.models[x](image)

    train_dict = dm.train_dataloader()
    assert len(train_dict) == len(years)

def test_fit(config, dm):
    years = dm.train.tile_year.unique()    
    m  = year.YearEnsemble(years=years, classes=dm.num_classes, label_dict=dm.species_label_dict, config=config)
    
    trainer = Trainer(fast_dev_run=False, max_epochs=1)
    trainer.fit(m, datamodule=dm)
    
def test_predict(config, dm):
    years = dm.train.tile_year.unique()     
    m  = year.YearEnsemble(years=years, classes=dm.num_classes, label_dict=dm.species_label_dict, config=config)
    trainer = Trainer(fast_dev_run=True)
    dls = dm.predict_dataloader(df=dm.test)
    predictions = trainer.predict(m, dataloaders=dls)
    ensemble_df = m.ensemble(predictions)
    ensemble_df.shape[0] == dm.test.shape[0]
    ensemble_df["individualID"] = ensemble_df["individual"]
    ensemble_df = ensemble_df.merge(dm.test, on="individualID")
    m.ensemble_metrics(ensemble_df)
