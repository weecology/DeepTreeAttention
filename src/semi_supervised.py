#Prepare semi_supervised data
import copy
import glob
import torch
from src.models import multi_stage
from src.data import TreeDataset
from pytorch_lightning import Trainer

def prepare_train_dataloader(config):
    """Generate a pytorch dataloader from unlabeled crop data"""
    semi_supervised_crops_csvs = glob.glob("{}/*.csv".format(config["semi_supervised"]["crop_dir"]))
    semi_supervised_crops = pd.concat([pd.read_csv(x) for x in semi_supervised_crops_csvs])
    if config["semi_supervised"]["site_filter"] is None:
        site_semi_supervised_crops = semi_supervised_crops[semi_supervised_crops.image_path.str.contains(config["semi_supervised"]["site_filter"])]
    else:
        site_semi_supervised_crops = semi_supervised_crops
    
    site_semi_supervised_crops = site_semi_supervised_crops.head(config["semi_supervised"]["num_samples"])
    
    #Predict labels for each crop
    config = copy.deepcopy(config)
    config["crop_dir"] = config["semi_supervised"]["crop_dir"]
    
    m = multi_stage.MultiStage.load_from_checkpoint(config["semi_supervised"]["model_path"], config=config)
    trainer = Trainer(gpus=config["gpus"], logger=False, enable_checkpointing=False)
    ds = TreeDataset(df=site_semi_supervised_crops, train=False, config=config)
    predictions = trainer.predict(m, dataloaders=m.predict_dataloader(ds))
                
    #Sample selection
    semi_supervised_ds = TreeDataset(df=site_semi_supervised_crops, train=False, config=config)
    data_loader = torch.utils.data.DataLoader(
        semi_supervised_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["workers"]
    )    
    
    return data_loader