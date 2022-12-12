#Test multi_stage
import numpy as np
import math
from pytorch_lightning import Trainer
from src.models import multi_stage
from src.data import TreeDataset
import pytest

@pytest.fixture()
def m(dm, config, ROOT):    
    taxonomic_csv = "{}/data/raw/families.csv".format(ROOT)        
    m  = multi_stage.MultiStage(train_df=dm.train, test_df=dm.train, config=config, taxonomic_csv=taxonomic_csv, debug=True)
    
    return m

def test_load(config, dm, m, ROOT, tmpdir):
    taxonomic_csv = "{}/data/raw/families.csv".format(ROOT)    
    m  = multi_stage.MultiStage(train_df=dm.train, test_df=dm.train, config=config, taxonomic_csv=taxonomic_csv, debug=False)    
    
    for index, level in enumerate(m.train_dataframes):
        len(m.label_to_taxonIDs[index].keys()) == len(level.label.unique())
        len(m.level_label_dicts[index].keys()) == len(level.label.unique())

    for index, level in enumerate(m.test_dataframes):
        len(m.label_to_taxonIDs[index].keys()) == len(level.label.unique())
        len(m.level_label_dicts[index].keys()) == len(level.label.unique())
    
    
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(m)
    
    #Confirm the model can be reloaded
    trainer.save_checkpoint("{}/test_model.pl".format(tmpdir))
    m2 = multi_stage.MultiStage.load_from_checkpoint("{}/test_model.pl".format(tmpdir))

def test_fit(config, m):
    trainer = Trainer(fast_dev_run=False, max_epochs=1, limit_train_batches=1, enable_checkpointing=False, num_sanity_val_steps=1)
    
    #Model can be trained and validated
    trainer.fit(m)
    metrics = trainer.validate(m)
    
    #Assert that the decision function from level 0 to level 1 is not NaN
    assert not math.isnan(metrics[0]["accuracy_CONIFER"])
    
def test_gather_predictions(config, dm, m, experiment):
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
    
    for site in ensemble_df.siteID.unique():
        site_result = ensemble_df[ensemble_df.siteID==site]
        combined_species = np.unique(site_result[['taxonID', 'ensembleTaxonID']].values)
        site_labels = {value:key for key, value in enumerate(combined_species)}
        y = [site_labels[x] for x in site_result.taxonID.values]
        ypred = [site_labels[x] for x in site_result.ensembleTaxonID.values]
        taxonlabels = [key for key, value in site_labels.items()]
        experiment.log_confusion_matrix(
            y,
            ypred,
            labels=taxonlabels,
            max_categories=len(taxonlabels),
            file_name="{}.json".format(site),
            title=site
        )
