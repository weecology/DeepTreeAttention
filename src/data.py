#Ligthning data module
from . import __file__
from distributed import as_completed
import glob
import geopandas as gpd
import numpy as np
import os
import pandas as pd
from pytorch_lightning import LightningDataModule
from src import generate
from src import CHM
from shapely.geometry import Point
import torch
from torch.utils.data import Dataset

class TreeDataset(Dataset):
    def __init__(self, csv_file):
        pass

import yaml
        
def filter_data(path, config):
    """Transform raw NEON data into clean shapefile   
    Args:
        config: DeepTreeAttention config dict, see config.yml
    """
    field = pd.read_csv(path)
    field = field[~field.elevation.isnull()]
    field = field[~field.growthForm.isin(["liana","small shrub"])]
    field = field[~field.growthForm.isnull()]
    field = field[~field.plantStatus.isnull()]        
    field = field[field.plantStatus.str.contains("Live")]    
    
    groups = field.groupby("individualID")
    shaded_ids = []
    for name, group in groups:
        shaded = any([x in ["Full shade", "Mostly shaded"] for x in group.canopyPosition.values])
        if shaded:
            if any([x in ["Open grown", "Full sun"] for x in group.canopyPosition.values]):
                continue
            else:
                shaded_ids.append(group.individualID.unique()[0])
        
    field = field[~(field.individualID.isin(shaded_ids))]
    field = field[(field.height > 3) | (field.height.isnull())]
    field = field[field.stemDiameter > config["min_stem_diameter"]]
    field = field[~field.taxonID.isin(["BETUL", "FRAXI", "HALES", "PICEA", "PINUS", "QUERC", "ULMUS", "2PLANT"])]
    field = field[~(field.eventID.str.contains("2014"))]
    with_heights = field[~field.height.isnull()]
    with_heights = with_heights.loc[with_heights.groupby('individualID')['height'].idxmax()]
    
    missing_heights = field[field.height.isnull()]
    missing_heights = missing_heights[~missing_heights.individualID.isin(with_heights.individualID)]
    missing_heights = missing_heights.groupby("individualID").apply(lambda x: x.sort_values(["eventID"],ascending=False).head(1)).reset_index(drop=True)
  
    field = pd.concat([with_heights,missing_heights])
    
    #remove multibole
    field = field[~(field.individualID.str.contains('[A-Z]$',regex=True))]

    #List of hand cleaned errors
    known_errors = ["NEON.PLA.D03.OSBS.03422","NEON.PLA.D03.OSBS.03422","NEON.PLA.D03.OSBS.03382", "NEON.PLA.D17.TEAK.01883"]
    field = field[~(field.individualID.isin(known_errors))]
    field = field[~(field.plotID == "SOAP_054")]
    
    #Create shapefile
    field["geometry"] = [Point(x,y) for x,y in zip(field["itcEasting"], field["itcNorthing"])]
    shp = gpd.GeoDataFrame(field)
    
    #HOTFIX, BLAN has some data in 18N UTM, reproject to 17N update columns
    BLAN_errors = shp[(shp.siteID == "BLAN") & (shp.utmZone == "18N")]
    BLAN_errors.set_crs(epsg=32618, inplace=True)
    BLAN_errors.to_crs(32617,inplace=True)
    BLAN_errors["utmZone"] = "17N"
    BLAN_errors["itcEasting"] = BLAN_errors.geometry.apply(lambda x: x.coords[0][0])
    BLAN_errors["itcNorthing"] = BLAN_errors.geometry.apply(lambda x: x.coords[0][1])
    
    #reupdate
    shp.loc[BLAN_errors.index] = BLAN_errors
    
    #Oak Right Lab has no AOP data
    shp = shp[~(shp.siteID.isin(["PUUM","ORNL"]))]

    return shp

def sample_plots(shp, test_fraction=0.1, min_samples=5):
    """Sample and split a pandas dataframe based on plotID
    Args:
        shp: pandas dataframe of filtered tree locations
        test_fraction: proportion of plots in test datasets
        min_samples: minimum number of samples per class
    """
    #split by plot level
    test_plots = shp.plotID.drop_duplicates().sample(frac=test_fraction)
    
    test = shp[shp.plotID.isin(test_plots)]
    train = shp[~shp.plotID.isin(test_plots)]
    
    test = test.groupby("taxonID").filter(lambda x: x.shape[0] > min_samples)
    
    train = train[train.taxonID.isin(test.taxonID)]
    test = test[test.taxonID.isin(train.taxonID)]
    
    return train, test
    
def train_test_split(shp, savedir, config, client = None, regenerate=False):
    """Create the train test split
    Args:
        shp: a filter pandas dataframe (or geodataframe)  
        savedir: directly to save train/test and metadata csv files
        client: optional dask client
        regenerate: recreate the train_test split
    Returns:
        None: train.shp and test.shp are written as side effect
        """    
    #set seed.
    np.random.seed(1)
    
    if regenerate:     
        most_species = 0
        if client:
            futures = [ ]
            for x in np.arange(config["iterations"]):
                future = client.submit(sample_plots, shp=shp, min_samples=config["min_samples"], test_fraction=config["test_fraction"])
                futures.append(future)
            
            for x in as_completed(futures):
                train, test = x.result()
                if len(train.taxonID.unique()) > most_species:
                    print(len(train.taxonID.unique()))
                    saved_train = train
                    saved_test = test
                    most_species = len(train.taxonID.unique())            
        else:
            for x in np.arange(config["iterations"]):
                train, test = sample_plots(shp)
                if len(train.taxonID.unique()) > most_species:
                    print(len(train.taxonID.unique()))
                    saved_train = train
                    saved_test = test
                    most_species = len(train.taxonID.unique())
        
        train = saved_train
        test = saved_test
    else:
        try:
            test_plots = gpd.read_file("{}/test.shp".format(savedir)).plotID.unique()
        except:
            raise FileNotFoundError("regenerate is {}, but {} is not found".format(regenerate, "{}/test.shp".format(savedir)))
        test = shp[shp.plotID.isin(test_plots)]
        train = shp[~shp.plotID.isin(test_plots)]
                
        train = train[train.taxonID.isin(test.taxonID)]
        test = test[test.taxonID.isin(train.taxonID)]
    
        ### This criteria was in the original repo, but I don't see the need Aug/9/2021
        #remove any test species that don't have site distributions in train
        ##to_remove = []
        ##for index,row in test.iterrows():
            ##if train[(train.taxonID==row["taxonID"]) & (train.siteID==row["siteID"])].empty:
                ##to_remove.append(index)
            
        #add_to_train = test[test.index.isin(to_remove)]
        #train = pd.concat([train, add_to_train])
        #test = test[~test.index.isin(to_remove)]    
        
        train = train[train.taxonID.isin(test.taxonID)]
        test = test[test.taxonID.isin(train.taxonID)]        
    
    print("There are {} records for {} species for {} sites in filtered train".format(
        train.shape[0],
        len(train.taxonID.unique()),
        len(train.siteID.unique())
    ))
    
    print("There are {} records for {} species for {} sites in test".format(
        test.shape[0],
        len(test.taxonID.unique()),
        len(test.siteID.unique())
    ))
    
    #Give tests a unique index to match against
    test["point_id"] = test.index.values
    train["point_id"] = train.index.values
    
    return train, test
        
def read_config(config_path):
    """Read config yaml file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

    except Exception as e:
        raise FileNotFoundError("There is no config at {}, yields {}".format(
            config_path, e))

    return config

#Dataset class
class TreeDataset(Dataset):
    def __init__(self, csv_file):
        pass

class TreeData(LightningDataModule):
    """
    Lightning data module to convert raw NEON data into HSI pixel crops based on the config.yml file. 
    The module checkpoints the different phases of setup, if one stage failed it will restart from that stage. 
    Use regenerate=True to override this behavior in setup()
    """
    def __init__(self, config=None, data_dir=None):
        super().__init__()
        self.ROOT = os.path.dirname(os.path.dirname(__file__))
        if data_dir is None:
            self.data_dir = "{}/data/".format(self.ROOT)
        else:
            self.data_dir = data_dir            
            
        if config is None:
            self.config = read_config("{}/config.yml".format(self.ROOT))   
        else:
            self.config = config
    
    def setup(self, csv_file, regenerate = False, client = None):
        #Clean data from raw csv, regenerate from scratch or check for progress and complete
        if regenerate:
            #remove any previous runs
            try:
                os.remove("{}/processed/test_points.shp".format(self.data_dir))
                os.remove(" ".format(self.data_dir))
                os.remove("{}/processed/filtered_data.csv".format(self.data_dir))
                os.remove("{}/processed/train_crowns.shp".format(self.data_dir))
                for x in glob.glob(self.config["crop_dir"]):
                    os.remove(x)
            except:
                pass
                
            #Convert raw neon data to x,y tree locatins
            df = filter_data(csv_file, config=self.config)
            #Filter points based on LiDAR height
            df = CHM.filter_CHM(df, CHM_pool=self.config["CHM_pool"],min_CHM_diff=self.config["min_CHM_diff"], min_CHM_height=self.config["min_CHM_height"])      
            df = df.groupby("taxonID").filter(lambda x: x.shape[0] > self.config["min_samples"])
            train, test = train_test_split(df,savedir="{}/processed".format(self.data_dir),config=self.config, regenerate=regenerate)   
            
            test.to_file("{}/processed/test_points.shp".format(self.data_dir))
            train.to_file("{}/processed/train_points.shp".format(self.data_dir))
            
            #Store class labels
            unique_species_labels = np.concatenate([train.taxonID.unique(), test.taxonID.unique()])
            unique_species_labels = np.unique(unique_species_labels)
            
            self.species_label_dict = {}
            for index, label in enumerate(unique_species_labels):
                self.species_label_dict[label] = index
        
            #test data 
            train_crowns = generate.points_to_crowns(
                field_data="{}/processed/train_points.shp".format(self.data_dir),
                rgb_dir=self.config["rgb_sensor_pool"],
                savedir=self.config["crop_dir"],
                raw_box_savedir=self.config["crop_dir"], 
                client=client
            )
            
            train_annotations = generate.generate_crops(train_crowns, savedir=self.config["crop_dir"])            
            train_annotations.to_csv("{}/processed/test.csv".format(self.data_dir))
            
            test_crowns = generate.points_to_crowns(
                field_data="{}/processed/test_points.shp".format(self.data_dir),
                rgb_dir=self.config["rgb_sensor_pool"],
                savedir=self.config["crop_dir"],
                raw_box_savedir=self.config["crop_dir"], 
                client=client
            )
        
            test_annotations = generate.generate_crops(test_crowns, savedir=self.config["crop_dir"])            
            test_annotations.to_csv("{}/processed/test.csv".format(self.data_dir))

    def train_dataloader(self):
        ds = TreeDataset(csv_file = "{}/processed/train.csv".format(self.ROOT))
        data_loader = torch.utils.data.DataLoader(
            ds,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config["workers"],
        )
        
        return data_loader
    
    def val_dataloader(self):
        ds = TreeDataset(csv_file = "{}/processed/test.csv".format(self.ROOT))
        data_loader = torch.utils.data.DataLoader(
            ds,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config["workers"],
        )
        
        return data_loader
