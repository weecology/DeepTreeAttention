#test baseline
from pytorch_lightning import Trainer
from src.models import baseline, Hang2020
import torch
import pytest

@pytest.fixture()
def m(dm, config): 
    config["lr"] = 0.01
    model = Hang2020.vanilla_CNN(bands=config["bands"], classes=dm.num_classes) 
    m = baseline.TreeModel(
        model=model, 
        config=config,
        classes=dm.num_classes, 
        loss_weight=torch.ones(dm.num_classes),
        label_dict=dm.species_label_dict)
    
    return m

def test_fit(dm, m):
    trainer = Trainer(fast_dev_run=False, max_epochs=1, limit_train_batches=1, enable_checkpointing=False, num_sanity_val_steps=0)    
    trainer.fit(m, datamodule=dm)