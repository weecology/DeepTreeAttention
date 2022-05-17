#Test multi_stage
from pytorch_lightning import Trainer
from src.models import multi_stage
from src import visualize
import torch
import pandas as pd
import numpy as np

def test_MultiStage(dm, config):
    m  = multi_stage.MultiStage(train_df=dm.train, test_df=dm.test,crowns=dm.crowns, config=config)
    image = torch.randn(20, 3, 110, 110)    
    for x in range(5):
        with torch.no_grad(): 
            output = m.models[x].model(image)
    
    train_dict = m.train_dataloader()
    assert len(train_dict) == 10
    
def test_fit(config, dm, comet_logger):
    m  = multi_stage.MultiStage(train_df=dm.train, test_df=dm.test, crowns=dm.crowns, config=config)
    trainer = Trainer(fast_dev_run=False, logger=comet_logger, num_sanity_val_steps=0, max_epochs=1)
    trainer.fit(m)
    
def test_predict(config, dm):
    m  = multi_stage.MultiStage(train_df=dm.train, test_df=dm.test, crowns=dm.crowns, config=config)
    trainer = Trainer(fast_dev_run=True)
    dls = m.predict_dataloader(df=dm.test)
    predictions = trainer.predict(m, dataloaders=dls)
    assert len(predictions) == 5

def test_gather_predictions(config, dm, comet_logger):
    m  = multi_stage.MultiStage(train_df=dm.train, test_df=dm.test, crowns=dm.crowns, config=config)
    trainer = Trainer(fast_dev_run=True)
    predictions = trainer.predict(m, dataloaders=m.predict_dataloader(df=dm.test))
    predictions = m.gather_predictions(predict_df=predictions, crowns=dm.crowns)    
    predictions.shape[0] == config["batch_size"]
    ensemble_df = m.ensemble(predictions)
    ensemble_df = m.evaluation_scores(ensemble_df, experiment=comet_logger.experiment)
    ensemble_df["pred_taxa_top1"] = ensemble_df.ensembleTaxonID
    ensemble_df["pred_label_top1"] = ensemble_df.ens_label    
    visualize.confusion_matrix(
        comet_experiment=comet_logger.experiment,
        results=ensemble_df,
        species_label_dict=dm.species_label_dict,
        test_crowns=dm.crowns,
        test=dm.test,
        test_points=dm.canopy_points,
        rgb_pool=None
    )
    
