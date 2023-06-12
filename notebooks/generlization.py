#Generalization across sites
from pytorch_lightning import Trainer
import pandas as pd
from pytorch_lightning.loggers import CometLogger
from src.data import config
from src.models import multi_stage

config = config("../config.yml")
comet_logger = CometLogger(project_name="DeepTreeAttention2", workspace=config["comet_workspace"], auto_output_logging="simple") 
comet_logger.experiment.add_tag("Generalization")
HARV_model = multi_stage.MultiStage.load_from_checkpoint("")
BART_test = pd.read_csv("/blue/ewhite/b.weinstein/DeepTreeAttention/d75360fc4261b0789df125b5f82930dd2fde0eb5/test_6a35fa16fbe649169efa826e2e7d3b6d_['BART'].csv")
BART_train = pd.read_csv("/blue/ewhite/b.weinstein/DeepTreeAttention/d75360fc4261b0789df125b5f82930dd2fde0eb5/test_6a35fa16fbe649169efa826e2e7d3b6d_['BART'].csv")

def finetune(m, n, test, train):
    """
    Args:
    m: a DeepTreeAttention model
    n: number of samples per class to take from train for finetune
    """
    trainer = Trainer()
    if n > 0:
        ids_to_keep = train.groupby("individual").apply(lambda x: x.head(1)).groupby("taxonID").apply(lambda x: x.head(n))
        filtered_train = train[train.individual.isin(ids_to_keep)]
        ds = multi_stage.TreeDataset(df=filtered_train, train=False, config=config)
        trainer.fit(m)
        
    ds = multi_stage.TreeDataset(df=test, train=False, config=config)
    predictions = trainer.predict(m, dataloaders=m.predict_dataloader(ds))
    results = m.gather_predictions(predictions)
    results = results.merge(BART_test[["individual","taxonID","label","siteID"]], on="individual")
    comet_logger.experiment.log_table("nested_predictions.csv", results)
    
    ensemble_df = m.ensemble(results)
    ensemble_df = m.evaluation_scores(
        ensemble_df,
        experiment=comet_logger.experiment
    )
    micro_accuracy = comet_logger.experiment.get_parameter("overall_micro")
    macro_accuracy = comet_logger.experiment.get_parameter("macro_accuracy")
    
    
