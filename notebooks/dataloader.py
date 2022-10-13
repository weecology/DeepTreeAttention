from src import data
from src import main
from src.models import Hang2020
import numpy as np
import pandas as pd

config = data.read_config("config.yml")
data_module = data.TreeData(csv_file="data/raw/neon_vst_data_2022.csv", client=None, metadata=True, comet_logger=None)
data_module.setup()
model = Hang2020.Hang2020(bands=config["bands"], classes=data_module.num_classes)
m = main.TreeModel(
    model=model, 
    classes=data_module.num_classes, 
    label_dict=data_module.species_label_dict)

dl = data_module.val_dataloader()

labels = []
for batch in dl:
    ind, image, label = batch
    labels.append(label)

labels = np.concatenate(labels)
taxonID = [data_module.label_to_taxonID[x] for x in labels]
pd.DataFrame(taxonID).value_counts()
    

