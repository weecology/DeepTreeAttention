#Test multi_stage
from pytorch_lightning import Trainer
from src.models import multi_stage
from src.data import TreeDataset
import pandas as pd
import numpy as np
from functools import reduce

def test_MultiStage(dm, config):
    m  = multi_stage.MultiStage(train_df=dm.train, test_df=dm.test,crowns=dm.crowns, config=config)
    
def test_fit(config, dm):
    m  = multi_stage.MultiStage(train_df=dm.train, test_df=dm.test, crowns=dm.crowns, config=config)
    trainer = Trainer(fast_dev_run=False, max_epochs=1)
    trainer.fit(m)

def test_gather_predictions(config, dm):
    m  = multi_stage.MultiStage(train_df=dm.train, test_df=dm.test, crowns=dm.crowns, config=config)
    trainer = Trainer(fast_dev_run=False)
    predict_datasets = []
    for level in range(m.levels):
        ds = TreeDataset(df=dm.test, train=False, config=config)
        predict_datasets.append(ds)

    predictions = trainer.predict(m, dataloaders=m.predict_dataloader(ds_list=predict_datasets))
    results = m.gather_predictions(predictions)
    assert len(np.unique(results.individual)) == len(np.unique(dm.test.individualID))
    
    results["individualID"] = results["individual"]
    results = results.merge(dm.test, on=["individualID"])
    assert len(np.unique(results.individual)) == len(np.unique(dm.test.individualID))
    
    ensemble_df = m.ensemble(results)
    ensemble_df = m.evaluation_scores(
        ensemble_df,
        experiment=None
    )    
    
    assert len(np.unique(ensemble_df.individualID)) == len(np.unique(dm.test.individualID))
