#test year model
from pytorch_lightning import Trainer
from src.models import year
import torch
import pandas as pd
import numpy as np

def test_YearEnsemble(dm, config):
    m = year.learned_ensemble(classes=dm.num_classes, config=config)
    images = [torch.randn(1, 349, 11, 11) for x in range(4)]
    mean_score = m(images)

    assert mean_score.shape == (1,dm.num_classes)