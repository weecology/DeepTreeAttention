# UNDE comparison, the goal is to look at the eval score of the old UNDE model against the test of the new UNDE model.
import comet_ml
from src.models import multi_stage
from src import visualize
from src import train
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CometLogger
from src.data import read_config
import os

config = read_config("config.yml")
config["crop_dir"] = os.path.join(config["data_dir"], config["use_data_commit"])

# Original model
m = multi_stage.MultiStage.load_from_checkpoint("/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/185caf9f910c4fd3a7f5e470b6828090_UNDE.pt", config=config)



m = multi_stage.MultiStage.load_from_checkpoint("/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/185caf9f910c4fd3a7f5e470b6828090_UNDE.pt", config=config)
comet_logger = CometLogger(project_name="DeepTreeAttention2", workspace=config["comet_workspace"], auto_output_logging="simple")     
comet_logger.experiment.add_tag("with swenson data")
trainer = Trainer(
            fast_dev_run=config["fast_dev_run"],
            max_epochs=config["epochs"],
            accelerator=config["accelerator"],
            num_sanity_val_steps=0,
            check_val_every_n_epoch=config["validation_interval"],
            enable_checkpointing=False,
            devices=1,
            logger=comet_logger)
ds = multi_stage.TreeDataset(df=m.test_df, train=False, config=config)    
predictions = trainer.predict(m, dataloaders=m.predict_dataloader(ds))
results = m.gather_predictions(predictions)
results = results.merge(m.test_df[["individual","taxonID","label","siteID","RGB_tile"]], on="individual")
comet_logger.experiment.log_table("nested_predictions.csv", results)
ensemble_df = m.ensemble(results)
ensemble_df = m.evaluation_scores(
    ensemble_df,
    experiment=comet_logger.experiment
)
comet_logger.experiment.end()

# hard copy that test dataset
new_test_dataset = m.test_df.copy(deep=True)

# Remove new sampled data
comet_logger = CometLogger(project_name="DeepTreeAttention2", workspace=config["comet_workspace"], auto_output_logging="simple")     
comet_logger.experiment.add_tag("without swenson data")
without_new_data = m.train_df[~(m.train_df.filename=="UNDE_Swenson")]
without_new_data = without_new_data.groupby("taxonID").filter(lambda x: x.shape[0] >= config["min_train_samples"])
new_test_dataset = new_test_dataset.groupby("taxonID").filter(lambda x: x.shape[0] >= config["min_test_samples"])
old_test_dataset = new_test_dataset[new_test_dataset.taxonID.isin(without_new_data.taxonID.unique())]
without_new_data = without_new_data[without_new_data.taxonID.isin(old_test_dataset.taxonID.unique())]
species_label_dict = {value:key for key, value in enumerate(old_test_dataset.taxonID.unique())}
old_test_dataset["label"] = old_test_dataset.taxonID.apply(lambda x: species_label_dict[x])
without_new_data["label"] = without_new_data.taxonID.apply(lambda x: species_label_dict[x])

config["pretrain_state_dict"] = "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/609ea91e99fd4e31ac56d384eb3af877_state_dict.pt"
m = multi_stage.MultiStage(config=config, train_df=without_new_data, test_df=old_test_dataset)
m.setup("fit")

for key in m.level_names:
    trainer = Trainer(
            fast_dev_run=False,
            max_epochs=config["epochs"],
            accelerator=config["accelerator"],
            num_sanity_val_steps=0,
            check_val_every_n_epoch=config["validation_interval"],
            enable_checkpointing=False,
            devices=1,
            logger=comet_logger)
    m.current_level = key
    m.configure_optimizers()
    trainer.fit(m)

    ds = multi_stage.TreeDataset(df=m.test_df, train=False, config=config)    
    predictions = trainer.predict(m, dataloaders=m.predict_dataloader(ds))
    results = m.gather_predictions(predictions)
    results = results.merge(m.test_df[["individual","taxonID","label","siteID","RGB_tile"]], on="individual")
    comet_logger.experiment.log_table("nested_predictions.csv", results)
    ensemble_df = m.ensemble(results)
    ensemble_df = m.evaluation_scores(
        ensemble_df,
        experiment=comet_logger.experiment
    )