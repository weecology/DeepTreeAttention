#Test multi_stage
from pytorch_lightning import Trainer
from src.models import multi_stage
from src import visualize
import torch
import pandas as pd
import numpy as np
from functools import reduce

def test_MultiStage(dm, config):
    m  = multi_stage.MultiStage(train_df=dm.train, test_df=dm.test,crowns=dm.crowns, config=config)
    image = torch.randn(20, 349, 110, 110)    
    for x in range(5):
        with torch.no_grad(): 
            output = m.models[x].model(image)
    
    train_dict = m.train_dataloader()
    assert len(train_dict) == 20
    
def test_fit(config, dm):
    m  = multi_stage.MultiStage(train_df=dm.train, test_df=dm.test, crowns=dm.crowns, config=config)
    trainer = Trainer(fast_dev_run=False, max_epochs=1)
    trainer.fit(m)

def test_gather_predictions(config, dm):
    m  = multi_stage.MultiStage(train_df=dm.train, test_df=dm.test, crowns=dm.crowns, config=config)
    trainer = Trainer(fast_dev_run=False)
    predictions = trainer.predict(m, dataloaders=m.val_dataloader())
    results = m.gather_levels(predictions)    
    results["individualID"] = results["individual"]
    results = results.merge(dm.test, on=["individualID"])
    ensemble_df = m.ensemble(results)
    ensemble_df = m.evaluation_scores(
        ensemble_df
    )    
