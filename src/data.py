#Ligthning data module
import glob
import geopandas as gpd
import numpy as np
import os
import pandas as pd
from pytorch_lightning import LightningDataModule
from src import generate
from . import __file__
from src import dataset 
from src import start_cluster
from src import CHM
import torch
import yaml

        
def filter_data(path, config):
    """Transform raw NEON data into clean shapefile   
    Args:
        config: DeepTreeAttention config dict, see config.yml
    """
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

def train_test_split(savedir, config, debug=False, client = None, regenerate=False):
    """Create the train test split
    Args:
        ROOT: 
        lookup_glob: The recursive glob path for the canopy height models to create a pool of .tif to search
        min_diff: minimum height diff between field and CHM data
        client: optional dask client
        """

    #atleast 10 data samples overall
    
    #set seed.
    np.random.seed(1)
    
    #TODO make regenerate flag.
    if regenerate:     
        most_species = 0
        if debug:
            iterations = 1
        else:
            iterations = 500
        
        if client:
            futures = [ ]
            for x in np.arange(iterations):
                future = client.submit(sample_plots, shp=shp)
                futures.append(future)
            
            for x in as_completed(futures):
                train, test = x.result()
                if len(train.taxonID.unique()) > most_species:
                    print(len(train.taxonID.unique()))
                    saved_train = train
                    saved_test = test
                    most_species = len(train.taxonID.unique())            
        else:
            for x in np.arange(iterations):
                train, test = sample_plots(shp)
                if len(train.taxonID.unique()) > most_species:
                    print(len(train.taxonID.unique()))
                    saved_train = train
                    saved_test = test
                    most_species = len(train.taxonID.unique())
        
        train = saved_train
        test = saved_test
    else:
        test_plots = gpd.read_file("{}/test.shp".format(ROOT)).plotID.unique()
        test = shp[shp.plotID.isin(test_plots)]
        train = shp[~shp.plotID.isin(test_plots)]
        
        test = test.groupby("taxonID").filter(lambda x: x.shape[0] > 5)
        
        train = train[train.taxonID.isin(test.taxonID)]
        test = test[test.taxonID.isin(train.taxonID)]
    
        #remove any test species that don't have site distributions in train
        to_remove = []
        for index,row in test.iterrows():
            if train[(train.taxonID==row["taxonID"]) & (train.siteID==row["siteID"])].empty:
                to_remove.append(index)
            
        add_to_train = test[test.index.isin(to_remove)]
        train = pd.concat([train, add_to_train])
        test = test[~test.index.isin(to_remove)]    
        
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
    
    #resample train
    if not n is None:
        train  =  train.groupby("taxonID").apply(lambda x: sample_if(x,n)).reset_index(drop=True)
            
    if not debug:    
        test.to_file("{}/test.shp".format(save_dir))
        train.to_file("{}/train.shp".format(save_dir))    
    
        #Create files for indexing
        #Create and save a new species and site label dict
        unique_species_labels = np.concatenate([train.taxonID.unique(), test.taxonID.unique()])
        unique_species_labels = np.unique(unique_species_labels)
        
        species_label_dict = {}
        for index, label in enumerate(unique_species_labels):
            species_label_dict[label] = index
        pd.DataFrame(species_label_dict.items(), columns=["taxonID","label"]).to_csv("{}/species_class_labels.csv".format(save_dir))    
        
        unique_site_labels = np.concatenate([train.siteID.unique(), test.siteID.unique()])
        unique_site_labels = np.unique(unique_site_labels)
        site_label_dict = {}
        for index, label in enumerate(unique_site_labels):
            site_label_dict[label] = index
        pd.DataFrame(site_label_dict.items(), columns=["siteID","label"]).to_csv("{}/data/processed/site_class_labels.csv".format(ROOT))  
        
        unique_domain_labels = np.concatenate([train.domainID.unique(), test.domainID.unique()])
        unique_domain_labels = np.unique(unique_domain_labels)
        domain_label_dict = {}
        for index, label in enumerate(unique_domain_labels):
            domain_label_dict[label] = index
        pd.DataFrame(domain_label_dict.items(), columns=["domainID","label"]).to_csv("{}/data/processed/domain_class_labels.csv".format(ROOT))  
        
def read_config(config_path):
    """Read config yaml file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

    except Exception as e:
        raise FileNotFoundError("There is no config at {}, yields {}".format(
            config_path, e))

    return config

class TreeData(LightningDataModule):
    def __init__(self):
        super().__init__()
        self.ROOT = os.path.dirname(os.path.dirname(__file__))
        self.data_dir = "{}/data/".format(self.ROOT)
        self.config = read_config("{}/config.yml".format(self.ROOT))        
    
    def setup(self, csv_file, regenerate = False):
        #Clean data from raw csv, regenerate from scratch or check for progress and complete
        if regenerate:
            #client = start_cluster.start(cpus=30)
            df = filter_data(csv_file, config=self.config)
            #Filter points based on LiDAR height
            df = CHM.filter_CHM(df, config=self.config)      
            df = df.groupby("taxonID").filter(lambda x: x.shape[0] > self.config["min_samples"])
            train, test = train_test_split(df,savedir="{}/processed".format(self.data_dir),config=self.config)   
            
            test.to_file("{}/processed/test_points.shp".format(self.data_dir))
            train.to_file("{}/processed/train_points.shp".format(self.data_dir))
            
            generate.points_to_crowns(
                field_data="{}/processed/test_points.shp".format(self.data_dir),
                rgb_dir=self.config["rgb_sensor_pool"],
                savedir=self.config["validation"]["crown_dir"],
                raw_box_savedir=self.config["crown_dir"],        
            )
                        
            generate.points_to_crowns(
                field_data="{}/processed/train_points.shp".format(self.data_dir),
                rgb_dir=self.config["rgb_sensor_pool"],
                savedir=self.config["train"]["crown_dir"],
                raw_box_savedir=self.config["crown_dir"],        
            )
            
            #For each shapefile, create crops and csv file
            train_crops = []
            for x in glob.glob("*.shp".format(self.config["train"]["crown_dir"])):
                crop_df = generate.generate_crops(x, savedir=self.config["crop_dir"])
                train_crops.append(crop_df)
                
            test_crops = []
            for x in glob.glob("*.shp".format(self.config["validation"]["crown_dir"])):
                crop_df = generate.generate_crops(x, savedir=self.config["crop_dir"])
                test_crops.append(crop_df)                
        if not os.path.exists("{}/processed/filtered_data.csv".format(self.data_dir)):
            filter_data()
        if not os.path.exists("{}/processed/train_points.shp".format(self.data_dir)):
            train_test_split(config=self.config)
        if not os.path.exists("{}/processed/train_crowns.shp".format(self.data_dir)):
            #test data
            generate.points_to_crowns(
                field_data="{}/processed/train_points.csv".format(self.data_dir),
                rgb_dir=self.config["rgb_sensor_pool"],
                savedir=self.config["crown_dir"],
                raw_box_savedir=self.config["crown_dir"],        
            )
                        
            generate.points_to_crowns(
                field_data="{}/processed/test_points.shp".format(self.data_dir),
                rgb_dir=self.config["rgb_sensor_pool"],
                savedir=self.config["crown_dir"],
                raw_box_savedir=self.config["crown_dir"],        
            )
            
        if len(glob.glob(self.config["crop_dir"])) == 0:
            generate.generate_crops(savedir=self.config["crop_dir"])            

    def train_dataloader(self):
        ds = dataset.TreeDataset(csv_file = "{}/processed/train.csv".format(self.ROOT))
        data_loader = torch.utils.data.DataLoader(
            ds,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config["workers"],
        )
        
        return data_loader
    
    def val_dataloader(self):
        ds = dataset.TreeDataset(csv_file = "{}/processed/test.csv".format(self.ROOT))
        data_loader = torch.utils.data.DataLoader(
            ds,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config["workers"],
        )
        
        return data_loader
