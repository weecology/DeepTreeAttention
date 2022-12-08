#test baseline
#Test multi_stage
import numpy as np
from pytorch_lightning import Trainer
from src.models import baseline, Hang2020
from src.data import TreeDataset
import math
import pytest

@pytest.fixture()
def m(dm, config, ROOT): 
    config["lr"] = 0.01
    model = Hang2020.vanilla_CNN(bands=config["bands"], classes=dm.num_classes) 
    m = baseline.TreeModel(
        model=model, 
        config=config,
        classes=dm.num_classes, 
        loss_weight=None,
        label_dict=dm.species_label_dict)
    
    
    return m

def test_fit(config, dm, m):
    trainer = Trainer(fast_dev_run=False, max_epochs=1, limit_train_batches=1, enable_checkpointing=False, num_sanity_val_steps=0)
    
    #Model can be trained and validated
    trainer.fit(m, datamodule=dm)
    
def test_evaluate_crowns(dm, m, experiment):
    results = m.evaluate_crowns(
        dm.val_dataloader(),
        crowns = dm.crowns,
        experiment=experiment,
    )
    
    assert results.shape[0] == dm.test.shape[0]