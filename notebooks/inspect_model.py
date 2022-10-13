#Inspect weights
from src import main
from src.data import TreeDataset
from src.data import read_config
from torch.utils.data import dataloader

config = read_config(config_path="../config.yml")
m = main.TreeModel.load_from_checkpoint("/Users/benweinstein/Downloads/b3022a3acdff4b5dbd338f0dff8fe969.pl")
config["crop_dir"] = "/Users/benweinstein/Downloads/b3022a3acdff4b5dbd338f0dff8fe969/"
ds = TreeDataset(csv_file="/Users/benweinstein/Downloads/b3022a3acdff4b5dbd338f0dff8fe969/train.csv", config=config)
dl = dataloader.DataLoader(ds, batch_size=3)
results = m.predict_dataloader(dl)