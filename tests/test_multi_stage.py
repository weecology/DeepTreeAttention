#Test multi_stage
from pytorch_lightning import Trainer
from src.models import multi_stage
from src.data import TreeDataset
from src import visualize 
import numpy as np
import glob

def test_MultiStage(dm, config):
    m  = multi_stage.MultiStage(train_df=dm.train, test_df=dm.train,crowns=dm.crowns, config=config)
    
def test_fit(config, dm, tmpdir):
    m  = multi_stage.MultiStage(train_df=dm.train, test_df=dm.test, crowns=dm.crowns, config=config)
    trainer = Trainer(fast_dev_run=True, limit_train_batches=1, profiler="simple")
    trainer.fit(m)
    
    trainer.save_checkpoint("{}/test_model.pl".format(tmpdir))
    config["preload_images"] = False
    m2 = multi_stage.MultiStage.load_from_checkpoint("{}/test_model.pl".format(tmpdir), config=config)
    assert m2.config["preload_images"] is False

def test_gather_predictions(config, dm, experiment):
    m  = multi_stage.MultiStage(train_df=dm.train, test_df=dm.test, crowns=dm.crowns, config=config)
    trainer = Trainer(fast_dev_run=False)
    ds = TreeDataset(df=dm.test, train=False, config=config)
    predictions = trainer.predict(m,m.predict_dataloader(ds))
    results = m.gather_predictions(predictions)
    assert len(np.unique(results.individual)) == len(np.unique(dm.test.individual))
    
    results["individual"] = results["individual"]
    results = results.merge(dm.test, on=["individual"])
    assert len(np.unique(results.individual)) == len(np.unique(dm.test.individual))
    
    ensemble_df = m.ensemble(results)
    ensemble_df = m.evaluation_scores(
        ensemble_df,
        experiment=None
    )    
    
    assert len(np.unique(ensemble_df.individual)) == len(np.unique(dm.test.individual))
    rgb_pool = glob.glob(dm.config["rgb_sensor_pool"], recursive=True)
    test = dm.test.groupby("individual").apply(lambda x: x.head(1)).reset_index(drop=True)
    visualize.confusion_matrix(
        comet_experiment=experiment,
        results=ensemble_df,
        species_label_dict=dm.species_label_dict,
        test_crowns=dm.crowns,
        test=test,
        test_points=dm.canopy_points,
        rgb_pool=rgb_pool,
        name="Test"
    )
