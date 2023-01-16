#Test semi_supervised.py
from pytorch_lightning import Trainer
from src import semi_supervised, predict
from src.models import joint_semi
import pytest
import numpy as np
import torch

@pytest.fixture()
def prediction_model_path(m, dm, config, tmpdir):
    trainer = Trainer(fast_dev_run=False, max_steps=1, limit_val_batches=1)
    trainer.fit(m)
    trainer.save_checkpoint("{}/checkpoint.pt".format(tmpdir))
    
    return "{}/checkpoint.pt".format(tmpdir)

def test_create_dataframe(config, m, dm, prediction_model_path):
    config["semi_supervised"]["crop_dir"] = config["crop_dir"]
    config["semi_supervised"]["model_path"] = prediction_model_path
    
    dm.train = semi_supervised.create_dataframe(config, unlabeled_df=dm.train, label_to_taxon_id=dm.label_to_taxonID)
    dm.train.shape[0] == config["semi_supervised"]["num_samples"]
    assert all(dm.train.ens_score > config["semi_supervised"]["threshold"])
    assert all(dm.train.index.values == np.arange(dm.train.shape[0]))
    
def test_fit(config, dm, prediction_model_path, ROOT, tmpdir):
    """Test that the model can load an existing model for unlabeled predictions and train """
    config["semi_supervised"]["site_filter"] = None
    config["semi_supervised"]["crop_dir"] = config["crop_dir"]
    config["semi_supervised"]["model_path"] = prediction_model_path
    dm.train.to_csv("{}/semi_supervised.csv".format(tmpdir))
    config["semi_supervised"]["semi_supervised_train"] = "{}/semi_supervised.csv".format(tmpdir)
    
    taxonomic_csv = "{}/data/raw/families.csv".format(ROOT)       
    m = joint_semi.TreeModel(train_df=dm.train, test_df=dm.test, taxonomic_csv=taxonomic_csv, config=config)

    trainer = Trainer(fast_dev_run=False, max_epochs=1, limit_train_batches=1, enable_checkpointing=False, num_sanity_val_steps=0)
    trainer.fit(m)
    assert not torch.isnan(trainer.logged_metrics["supervised_loss_level_0"])