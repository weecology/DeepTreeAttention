#Test multi_stage
import numpy as np
from pytorch_lightning import Trainer
from src.models import multi_stage
from src.models.multi_stage import TreeDataset
import pytest

@pytest.fixture()
def m(dm, config,ROOT):    
    m  = multi_stage.MultiStage(train_df=dm.train, test_df=dm.train, config=config, debug=True)
    m.setup("fit")    
    
    return m

@pytest.mark.parametrize("preload_images",[True, False])
def test_load(config, dm, m, tmpdir, preload_images):
    config["preload_images"] = preload_images
    m  = multi_stage.MultiStage(train_df=dm.train, test_df=dm.train, config=config, debug=False)    
    m.setup("fit")
    for index, level in m.train_dataframes.items():
        len(m.label_to_taxonIDs[index].keys()) == len(level.label.unique())
        len(m.level_label_dicts[index].keys()) == len(level.label.unique())

    for index, level in m.test_dataframes.items():
        len(m.label_to_taxonIDs[index].keys()) == len(level.label.unique())
        len(m.level_label_dicts[index].keys()) == len(level.label.unique())
    
    trainer = Trainer(fast_dev_run=True)
    for key in m.level_names:
        m.current_level = key
        m.configure_optimizers()
        trainer.fit(m)
        metrics = trainer.validate(m)
        
    #Confirm the model can be reloaded
    trainer.save_checkpoint("{}/test_model.pt".format(tmpdir))
    m2 = multi_stage.MultiStage.load_from_checkpoint("{}/test_model.pt".format(tmpdir), config=config)

def test_fit_no_conifer(config, dm):   
    one_conifer = dm.test[dm.test.taxonID.isin(["QULA2","QUEGE2","QUNI","MAGNO","LIST2","ACRU","NYSY","CAGL8","QUAL3"])]
    m  = multi_stage.MultiStage(train_df=one_conifer, test_df=one_conifer, config=config)
    m.setup("fit")
    trainer = Trainer(fast_dev_run=False, max_epochs=1, enable_checkpointing=False)
    
    #Model can be trained and validated
    for key in m.level_names:
        m.current_level = key
        m.configure_optimizers()
        trainer.fit(m)
        metrics = trainer.validate(m)
        
def test_fit_one_conifer(config, dm):   
    one_conifer = dm.test[dm.test.taxonID.isin(["QULA2","QUEGE2","QUNI","MAGNO","LIST2","ACRU","NYSY","CAGL8","QUAL3","PIPA2"])]
    m  = multi_stage.MultiStage(train_df=one_conifer, test_df=one_conifer, config=config)
    
    trainer = Trainer(fast_dev_run=False, max_epochs=1, enable_checkpointing=False)
    
    #Model can be trained and validated
    m.setup("fit")
    
    for key in m.level_names:
        m.current_level = key
        m.configure_optimizers()
        trainer.fit(m)
        metrics = trainer.validate(m)

def test_two_species(config, dm):   
    two_species = dm.test[dm.test.taxonID.isin(["MAGNO","LIST2"])]
    m  = multi_stage.MultiStage(train_df=two_species, test_df=two_species, config=config)
    
    trainer = Trainer(fast_dev_run=False, max_epochs=1, enable_checkpointing=False)
    
    #Model can be trained and validated
    m.setup("fit")
    
    for key in m.level_names:
        m.current_level = key
        m.configure_optimizers()
        trainer.fit(m)
        metrics = trainer.validate(m)
        
def test_fit(config, m):
    trainer = Trainer(fast_dev_run=False, max_epochs=1, limit_train_batches=1, enable_checkpointing=False, num_sanity_val_steps=1)
    
    #Model can be trained and validated
    for key in m.level_names:
        m.current_level = key
        m.configure_optimizers()
        trainer.fit(m)
        metrics = trainer.validate(m)   
        
def test_gather_predictions(config, dm, m):
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

def test_gather_predictions_no_conifer(config, dm, m):
    trainer = Trainer(fast_dev_run=True)
    no_conifer = dm.test[dm.test.taxonID.isin(["QULA2","QUEGE2","QUNI","MAGNO","LIST2","ACRU","NYSY","CAGL8","QUAL3"])]
    ds = TreeDataset(df=no_conifer, train=False, config=config)
    predictions = trainer.predict(m, dataloaders=m.predict_dataloader(ds))
    results = m.gather_predictions(predictions)
    assert len(np.unique(results.individual)) == len(np.unique(no_conifer.individual))
    
    results["individualID"] = results["individual"]
    results = results.merge(no_conifer, on=["individual"])
    assert len(np.unique(results.individual)) == len(np.unique(no_conifer.individual))
    
    ensemble_df = m.ensemble(results)
    ensemble_df = m.evaluation_scores(
        experiment=None,
        ensemble_df=ensemble_df)
    
    assert len(ensemble_df.taxonID.unique()) == len(no_conifer.taxonID.unique())
    
def test_gather_predictions_no_oak(config, dm, m):
    trainer = Trainer(fast_dev_run=True)
    no_oak = dm.test[dm.test.taxonID.isin(["MAGNO","LIST2","ACRU","NYSY","CAGL8"])]
    ds = TreeDataset(df=no_oak, train=False, config=config)
    predictions = trainer.predict(m, dataloaders=m.predict_dataloader(ds))
    results = m.gather_predictions(predictions)
    assert len(np.unique(results.individual)) == len(np.unique(no_oak.individual))
    
    results["individualID"] = results["individual"]
    results = results.merge(no_oak, on=["individual"])
    assert len(np.unique(results.individual)) == len(np.unique(no_oak.individual))
    
    ensemble_df = m.ensemble(results)
    ensemble_df = m.evaluation_scores(
        experiment=None,
        ensemble_df=ensemble_df)