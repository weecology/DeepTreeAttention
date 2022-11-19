#Test multi_stage
import numpy as np
from pytorch_lightning import Trainer
from src.models import multi_stage
from src.data import TreeDataset
import math

def test_MultiStage(dm, config):
    m  = multi_stage.MultiStage(train_df=dm.train, test_df=dm.train,crowns=dm.crowns, config=config, debug=True)

def test_reload(config, dm, tmpdir):
    config["preload_images"] = True    
    m  = multi_stage.MultiStage(train_df=dm.train, test_df=dm.test, crowns=dm.crowns, config=config, debug=False)
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(m)
    
    #Confirm the model can be reloaded
    trainer.save_checkpoint("{}/test_model.pl".format(tmpdir))
    m2 = multi_stage.MultiStage.load_from_checkpoint("{}/test_model.pl".format(tmpdir), config=config)

    
def test_fit(config, dm, tmpdir):
    config["preload_images"] = True    
    m  = multi_stage.MultiStage(train_df=dm.train, test_df=dm.test, crowns=dm.crowns, config=config, debug=True)
    trainer = Trainer(fast_dev_run=False, max_epochs=1, limit_train_batches=1, enable_checkpointing=False, num_sanity_val_steps=0)
    
    #Model can be trained and validated
    trainer.fit(m)
    metrics = trainer.validate(m)
    
    #Assert that the decision function from level 0 to level 1 is not NaN
    assert not math.isnan(metrics[0]["accuracy_OTHER_level_0"])
    
def test_gather_predictions(config, dm):
    m  = multi_stage.MultiStage(train_df=dm.train, test_df=dm.test, crowns=dm.crowns, config=config, debug=True)
    trainer = Trainer(fast_dev_run=True)
    ds = TreeDataset(df=dm.test, train=False, config=config)
    predictions = trainer.predict(m, dataloaders=m.predict_dataloader(ds))
    results = m.gather_predictions(predictions)
    assert len(np.unique(results.individual)) == len(np.unique(dm.test.individual))
    
    results["individualID"] = results["individual"]
    results = results.merge(dm.test, on=["individual"])
    assert len(np.unique(results.individual)) == len(np.unique(dm.test.individual))
    
    ensemble_df = m.ensemble(results)
    ensemble_df = m.evaluation_scores(
        ensemble_df,
        experiment=None
    )
