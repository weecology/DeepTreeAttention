#Test multi_stage
from pytorch_lightning import Trainer
from src.models import multi_stage
from src.data import TreeDataset
from src import visualize 
import numpy as np
import glob

def test_MultiStage(dm, config):
    m  = multi_stage.MultiStage(train_df=dm.train, test_df=dm.train,crowns=dm.crowns, config=config)

def test_fit(config, dm):
    m  = multi_stage.MultiStage(train_df=dm.train, test_df=dm.test, crowns=dm.crowns, config=config)
    trainer = Trainer(fast_dev_run=True, profiler="simple")
    trainer.fit(m)

def test_gather_predictions(config, dm, experiment):
    m  = multi_stage.MultiStage(train_df=dm.train, test_df=dm.train, crowns=dm.crowns, config=config)
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
    rgb_pool = glob.glob(dm.config["rgb_sensor_pool"], recursive=True)
    test = dm.test.groupby("individualID").apply(lambda x: x.head(1)).reset_index(drop=True)
    visualize.confusion_matrix(
        comet_experiment=experiment,
        results=ensemble_df,
        species_label_dict=dm.species_label_dict,
        test_crowns=dm.crowns,
        test=test,
        test_points=dm.canopy_points,
        rgb_pool=rgb_pool
    )
