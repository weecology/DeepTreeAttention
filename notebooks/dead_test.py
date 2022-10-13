from src.models import dead
from torchvision.datasets import ImageFolder

from src.data import read_config
import rasterio as rio
import numpy as np
import glob
import torch

config = read_config("config.yml")
m = dead.AliveDead.load_from_checkpoint(config["dead_model"], config=config)
m.eval()
transform = dead.get_transform(augment=False)
ds = ImageFolder(root="data/raw/dead_test/", transform=transform)
data_loader = torch.utils.data.DataLoader(
    ds,
    batch_size=10,
    shuffle=False,
    num_workers=0
)
labels = [] 
targets = []
for batch in data_loader: 
    x,y = batch    
    with torch.no_grad():
        pred = m(x)        
    labels.append(np.argmax(pred.numpy(),1))
    targets.append(y.numpy())

targets = np.concatenate(targets)
labels = np.concatenate(labels)

index = np.where(targets==1)[0]
labels = [labels[x] for x in index]
np.sum([x == 1 for x in labels])/len(labels)

imgs = glob.glob("data/raw/dead_test/Dead/*")
labels = []
for i in imgs:    
    data = rio.open(i).read()
    box = transform(np.rollaxis(data, 0,3))
    pred = m(box.unsqueeze(0))
    labels.append(np.argmax(pred.detach()))

np.sum([x == 1 for x in labels])/len(labels)


data = rio.open("tests/data/dead_tree.png").read()
box = transform(np.rollaxis(data, 0,3))
pred = m(box.unsqueeze(0))
labels.append(np.argmax(pred.detach()))
