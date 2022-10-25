#test dead
from src.models import dead
from pytorch_lightning import Trainer
import torch

def test_fit(config):
    trainer = Trainer(fast_dev_run=True)
    m = dead.AliveDead(config=config)
    trainer.fit(m)
    assert True

def test_validate(config):
    trainer = Trainer(fast_dev_run=True)
    m = dead.AliveDead(config=config)
    trainer.validate(m)
    assert True   