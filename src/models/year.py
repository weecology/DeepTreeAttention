#Linear year converter
#For a given HSI year, convert to 2021 by linear mapping
from pytorch_lightning import LightningModule
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
import rasterio as rio

class year_dataset(Dataset):
    """Yield a pair of pixels for linear regression
    Args:
        csv_file: path to a .csv file with image_path column
        band: band number to select from image matrix
        year: input year to convert to target_year
        target_year: year to convert into
    """
    def __init__(self, csv_file, band, year, target_year = 2019):
        self.annotations = pd.read_csv(csv_file)
        self.band = band
        self.individuals = self.annotations.individualID.unique()
        self.year = year
        self.target_year = target_year
    
    def __len__(self):
        return len(self.individuals)
    
    def __getitem__(self, index):
        individual = self.individuals[index]
        images = self.annotations.loc[self.annotations.individualID == individual, "image_path"]
        target_image = [x for x in images if str(self.target_year) in x] 
        input_image = [x for x in images if str(self.year) in x] 
        
        input_data = rio.open(input_image).read()
        input_data = input_data[self.band,:,:]
        target_data = rio.open(target_image).read()
        target_data = target_data[self.band,:,:]
        
        pixel_pairs = zip(target_data, input_data)
        
        return pixel_pairs
        
    
class year_converter(LightningModule):
    def __init__(self, csv_file, band, year, config):
        super(year_converter, self).__init__()
        self.csv_file = csv_file
        self.config = config
        self.band = band
        self.year = year
    
        #Linear 
        self.linear = torch.nn.Linear(1, 1) 
    
    def forward(self, x):
        x = self.linear(x)
        
        return x
    
    def training_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat,y)
        
        return loss
    
    def train_dataloader(self):
        self.train_ds = year_dataset(self.csv_file, self.band, self.year)
        data_loader = torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config["workers"],
        )
        
        return data_loader
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        
        return optimizer

    