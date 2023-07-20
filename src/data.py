#Ligthning data module
from . import __file__
from distributed import wait
import glob
import geopandas as gpd
import numpy as np
import os
import pandas as pd
from pytorch_lightning import LightningDataModule
from src import generate, CHM, augmentation, megaplot, neon_paths, sampler
from src.models import dead
from src.utils import *

from shapely.geometry import Point
import torch
from torch.utils.data import Dataset
import rasterio
import tarfile

# Dataset class
class TreeDataset(Dataset):
    """A csv file with a path to image crop and label
    Args:
       csv_file: path to csv file with image_path and label
       df: pandas dataframe
    """
    def __init__(self, csv_file=None, df=None, config=None, train=True):
        if df is None:
            self.annotations = pd.read_csv(csv_file)
        else:
            self.annotations = df
        self.train = train
        self.config = config         
        self.image_size = config["image_size"]

        # Create augmentor
        self.transformer = augmentation.train_augmentation(image_size=self.image_size)
                        
        # Pin data to memory if desired
        if self.config["preload_images"]:
            self.image_dict = {}
            for index, row in self.annotations.iterrows():
                image_path = os.path.join(self.config["crop_dir"],row["image_path"])
                self.image_dict[index] = load_image(image_path, image_size=self.image_size)
        

    def __len__(self):
        # 0th based index
        return self.annotations.shape[0]

    def __getitem__(self, index):
        inputs = {}
        image_path = self.annotations.image_path.iloc[index]      
        individual = os.path.basename(os.path.splitext(image_path)[0])
        if self.config["preload_images"]:
            inputs["HSI"] = self.image_dict[index]
        else:
            image_basename = self.annotations.image_path.iloc[index]  
            image_path = os.path.join(self.config["crop_dir"],image_basename)                
            image = load_image(image_path, image_size=self.image_size)
            inputs["HSI"] = image

        if self.train:
            label = self.annotations.label.iloc[index]
            label = torch.tensor(label, dtype=torch.long)
            inputs["HSI"] = self.transformer(inputs["HSI"])

            return individual, inputs, label
        else:
            return individual, inputs
        
def filter_data(path, config):
    """Transform raw NEON data into clean shapefile   
    Args:
        config: DeepTreeAttention config dict, see config.yml
    """
    raw_data = pd.read_csv(path)
    raw_data["individual"] = raw_data["individualID"]
    raw_data = raw_data[~raw_data.plantStatus.isnull()]  
    
    field = raw_data[~raw_data.itcEasting.isnull()]
    field = field[~field.growthForm.isin(["liana","small shrub"])]
    field = field[~field.growthForm.isnull()]
    field = field[~field.plantStatus.isnull()]        
    field = field[field.plantStatus.str.contains("Live")]    
    
    #Make sure there were not previously labeled dead
    is_dead = raw_data[raw_data.plantStatus.str.contains("Dead | dead")]    
    field = field[~field.individual.isin(is_dead.individual)]
    
    groups = field.groupby("individual")
    shaded_ids = []
    for name, group in groups:
        shaded = any([x in ["Full shade", "Mostly shaded"] for x in group.canopyPosition.values])
        if shaded:
            if any([x in ["Open grown", "Full sun"] for x in group.canopyPosition.values]):
                continue
            else:
                shaded_ids.append(group.individual.unique()[0])
        
    field = field[~(field.individual.isin(shaded_ids))]
    field = field[(field.height > 3) | (field.height.isnull())]
    field = field[field.stemDiameter > config["min_stem_diameter"]]
    
    #Subspecies filter    
    field.loc[field.taxonID=="PICOL","taxonID"] = "PICO"    
    field.loc[field.taxonID=="PSMEM","taxonID"] = "PSME"
    field.loc[field.taxonID=="ABLAL","taxonID"] = "ABLA"    
    field.loc[field.taxonID=="ACSAS","taxonID"] = "ACSA3"    
    field.loc[field.taxonID=="BEPAP","taxonID"] = "BEPA"
    field.loc[field.taxonID=="PIPOS","taxonID"] = "PIPO"    
    field.loc[field.taxonID=="ACNEN","taxonID"] = "ACNE2"
    field.loc[field.taxonID=="ACRUR","taxonID"] = "ACRU"
    field.loc[field.taxonID=="CECAC","taxonID"] = "CECA4"
    field.loc[field.taxonID=="PRSES","taxonID"] = "PRSE2"
    field.loc[field.taxonID=="BEPAC2","taxonID"] = "BEPA"
    field.loc[field.taxonID=="JUVIV","taxonID"] = "JUVI"
    field.loc[field.taxonID=="PRPEP","taxonID"] = "PRPE2"
    field.loc[field.taxonID=="COCOC","taxonID"] = "COCO6"
    field.loc[field.taxonID=="NYBI","taxonID"] = "NYSY"
    field.loc[field.taxonID=="ARVIM","taxonID"] = "ARVI4"
    
    field = field[~field.taxonRank.isin(["speciesGroup", "subspecies", "genus","kingdom"])]
    field = field[~(field.eventID.str.contains("2014"))]
    with_heights = field[~field.height.isnull()]
    with_heights = with_heights.loc[with_heights.groupby('individual')['height'].idxmax()]
    
    missing_heights = field[field.height.isnull()]
    missing_heights = missing_heights[~missing_heights.individual.isin(with_heights.individual)]
    missing_heights = missing_heights.groupby("individual").apply(lambda x: x.sort_values(["eventID"],ascending=False).head(1)).reset_index(drop=True)
  
    field = pd.concat([with_heights,missing_heights])
    
    # Remove multibole
    field = field[~(field.individual.str.contains('[A-Z]$',regex=True))]

    # List of hand cleaned errors
    known_errors = ["NEON.PLA.D03.OSBS.03422","NEON.PLA.D03.OSBS.03422","NEON.PLA.D03.OSBS.03382", "NEON.PLA.D17.TEAK.01883"]
    field = field[~(field.individual.isin(known_errors))]
    field = field[~(field.plotID == "SOAP_054")]
    
    #Create shapefile
    field["geometry"] = [Point(x,y) for x,y in zip(field["itcEasting"], field["itcNorthing"])]
    shp = gpd.GeoDataFrame(field)
    
    # BLAN has some data in 18N UTM, reproject to 17N update columns
    BLAN_errors = shp[(shp.siteID == "BLAN") & (shp.utmZone == "18N")]
    BLAN_errors.set_crs(epsg=32618, inplace=True)
    BLAN_errors.to_crs(32617,inplace=True)
    BLAN_errors["utmZone"] = "17N"
    BLAN_errors["itcEasting"] = BLAN_errors.geometry.apply(lambda x: x.coords[0][0])
    BLAN_errors["itcNorthing"] = BLAN_errors.geometry.apply(lambda x: x.coords[0][1])
    
    # reupdate
    shp.loc[BLAN_errors.index] = BLAN_errors
    
    # Oak Right Lab has no AOP data
    shp = shp[~(shp.siteID.isin(["PUUM","ORNL"]))]

    # There are a couple NEON plots within the OSBS megaplot, make sure they are removed
    shp = shp[~shp.plotID.isin(["OSBS_026","OSBS_029","OSBS_039","OSBS_027","OSBS_036"])]

    return shp


def sample_plots(shp, min_train_samples=5, min_test_samples=3, iteration = 1):
    """Sample and split a pandas dataframe based on plotID
    Args:
        shp: pandas dataframe of filtered tree locations
        test_fraction: proportion of plots in test datasets
        min_samples: minimum number of samples per class
        iteration: a dummy parameter to make dask submission unique
    """
    #split by plot level
    plotIDs = list(shp.plotID.unique())
    if len(plotIDs) <=2:
        test = shp[shp.plotID == shp.plotID.unique()[0]]
        train = shp[shp.plotID == shp.plotID.unique()[1]]

        return train, test
    else:
        plotIDs = shp.plotID.unique()

    np.random.shuffle(plotIDs)
    species_to_sample = shp.taxonID.unique()
    test_plots = []
    for plotID in plotIDs:
        selected_plot = shp[shp.plotID == plotID]
        # If any species is missing from min samples, include plot
        for x in selected_plot.taxonID.unique():
            if x in species_to_sample:
                test_plots.append(plotID)
                # Update species list                
                counts = shp[shp.plotID.isin(test_plots)].taxonID.value_counts()                
                species_completed = counts[counts > min_test_samples].index.tolist()
                species_to_sample = [x for x in shp.taxonID.unique() if not x in species_completed]
                
    test = shp[shp.plotID.isin(test_plots)]
    train = shp[~shp.plotID.isin(test.plotID.unique())]
    
    # Remove fixed boxes from test
    test = test.loc[~test["box_id"].astype(str).str.contains("fixed").fillna(False)]    
    test = test.groupby("taxonID").filter(lambda x: x.shape[0] >= min_test_samples)
    train = train.groupby("taxonID").filter(lambda x: x.shape[0] >= min_train_samples)
        
    train = train[train.taxonID.isin(test.taxonID)]    
    test = test[test.taxonID.isin(train.taxonID)]

    return train, test


def train_test_split(shp, config, client = None):
    """Create the train test split
    Args:
        shp: a filter pandas dataframe (or geodataframe)  
        client: optional dask client
    Returns:
        None: train.shp and test.shp are written as side effect
        """    
    min_sampled = config["min_train_samples"] + config["min_test_samples"]
    keep = shp.groupby("individual").apply(lambda x: x.head(1)).taxonID.value_counts() > (min_sampled)
    species_to_keep = keep[keep].index
    shp = shp[shp.taxonID.isin(species_to_keep)]
    print("splitting data into train test. Initial data has {} points from {} species with a min of {} samples".format(shp.shape[0],shp.taxonID.nunique(),min_sampled))
    test_species = 0
    ties = []
    if client:
        futures = [ ]
        for x in np.arange(config["iterations"]):
            future = client.submit(sample_plots, shp=shp, min_train_samples=config["min_train_samples"], iteration=x, min_test_samples=config["min_test_samples"])
            futures.append(future)
        
        wait(futures)
        for x in futures:
            train, test = x.result()
            if test.taxonID.nunique() > test_species:
                print("Selected test has {} points and {} species".format(test.shape[0], test.taxonID.nunique()))
                saved_train = train
                saved_test = test
                test_species = test.taxonID.nunique()
                ties = []
                ties.append([train, test])
            elif test.taxonID.nunique() == test_species:
                ties.append([train, test])          
    else:
        for x in np.arange(config["iterations"]):
            train, test = sample_plots(shp, min_train_samples=config["min_train_samples"], min_test_samples=config["min_test_samples"])
            if test.taxonID.nunique() > test_species:
                print("Selected test has {} points and {} species".format(test.shape[0], test.taxonID.nunique()))
                saved_train = train
                saved_test = test
                test_species = test.taxonID.nunique()
                #reset ties
                ties = []
                ties.append([train, test])
            elif test.taxonID.nunique() == test_species:
                ties.append([train, test])
    
    # The size of the datasets
    if len(ties) > 1:
        print("The size of tied datasets with {} species is {}".format(test_species, [x[1].shape[0] for x in ties]))        
        saved_train, saved_test = ties[np.argmin([x[1].shape[0] for x in ties])]
        
    train = saved_train
    test = saved_test    
    
    # Give tests a unique index to match against
    test["point_id"] = test.index.values
    train["point_id"] = train.index.values
    
    return train, test

def filter_dead_annotations(crowns, config):
    """Given a set of annotations, predict whether RGB is dead
    Args:
        annotations: must contain xmin, xmax, ymin, ymax and image path fields"""
    ds = dead.utm_dataset(crowns, config=config)
    dead_model = dead.AliveDead.load_from_checkpoint(config["dead_model"], config=config)    
    label, score = dead.predict_dead_dataloader(dead_model=dead_model, dataset=ds, config=config)
    
    return label, score
    
class TreeData(LightningDataModule):
    """
    Lightning data module to convert raw NEON data into HSI pixel crops based on the config.yml file. 
    The module checkpoints the different phases of setup, if one stage failed it will restart from that stage. 
    Use regenerate=True to override this behavior in setup()
    """
    def __init__(self, csv_file, config, experiment_id=None, comet_logger=None, client = None, data_dir=None, site=None, create_train_test=False, filter_species_site=None):
        """
        Args:
            config: optional config file to override
            data_dir: override data location, defaults to ROOT   
            regenerate: Whether to recreate raw data
            site: only use data from a siteID
            experiment_id: string to use to label train-test split
            comet_logger: pytorch comet_logger for comet_ml experiment
        """
        super().__init__()
        self.ROOT = os.path.dirname(os.path.dirname(__file__))
        self.csv_file = csv_file
        self.comet_logger = comet_logger
        self.site = site
        self.experiment_id = experiment_id
        self.create_train_test = create_train_test
        
        # Default training location
        self.client = client
        self.data_dir = data_dir
        self.config = config
        
        #add boxes folder if needed
        try:
            os.mkdir(os.path.join(self.data_dir,"boxes"))
        except:
            pass
        
        # Clean data from raw csv, regenerate from scratch or check for progress and complete                        
        if self.config["use_data_commit"] is None:
            self.rgb_pool, self.h5_pool, self.hsi_pool, self.CHM_pool = create_glob_lists(config)
            if self.config["replace_bounding_boxes"]: 
                # Convert raw neon data to x,y tree locatins
                df = filter_data(self.csv_file, config=self.config)
                if site:
                    if not site == "pretrain":
                        df = df[df.siteID.isin(site)]
                # Load any megaplot data
                if not self.config["megaplot_dir"] is None:
                    megaplot_data = megaplot.load(directory=self.config["megaplot_dir"], config=self.config, client=self.client, site=site)
                    megaplot_data.loc[megaplot_data.taxonID=="MAGR4","taxonID"] = "MAGNO"  

                    # Hold IFAS records seperarely to model on polygons
                    IFAS = megaplot_data[megaplot_data.filename.str.contains("IFAS")]
                    IFAS.geometry = IFAS.geometry.envelope
                    IFAS["box_id"] = list(range(IFAS.shape[0]))
                    IFAS = IFAS[["geometry","taxonID","individual","plotID","siteID","box_id"]]
                    IFAS["individual"] = IFAS["individual"]
                    megaplot_data = megaplot_data[~(megaplot_data.filename.str.contains("IFAS"))]
                    df = pd.concat([megaplot_data, df])
                    
                if self.comet_logger:
                    self.comet_logger.experiment.log_parameter("Species before CHM filter", len(df.taxonID.unique()))
                    self.comet_logger.experiment.log_parameter("Samples before CHM filter", df.shape[0])
                
                if filter_species_site:
                    species_to_keep = df[df.siteID.isin(filter_species_site)].taxonID.unique()
                    df = df[df.taxonID.isin(species_to_keep)]
                    
                df.to_file("{}/unfiltered_points_{}.shp".format(self.data_dir, site))
                
                #Filter points based on LiDAR height for NEON data
                self.canopy_points = CHM.filter_CHM(df, CHM_pool=self.CHM_pool,
                                    min_CHM_height=self.config["min_CHM_height"], 
                                    max_CHM_diff=self.config["max_CHM_diff"], 
                                    CHM_height_limit=self.config["CHM_height_limit"])  
                
                self.canopy_points.to_file("{}/canopy_points.shp".format(self.data_dir))

                if self.comet_logger:
                    self.comet_logger.experiment.log_parameter("Species after CHM filter", len(self.canopy_points .taxonID.unique()))
                    self.comet_logger.experiment.log_parameter("Samples after CHM filter", self.canopy_points.shape[0])
            
                # Create crown data on gpu
                self.crowns = generate.points_to_crowns(
                    field_data="{}/canopy_points.shp".format(self.data_dir),
                    rgb_pool=self.rgb_pool,
                    savedir="{}/boxes/".format(self.data_dir),
                    raw_box_savedir="{}/boxes/".format(self.data_dir),
                    client=None
                )
                
                if self.config["megaplot_dir"]:
                    #Add IFAS back in, use polygons instead of deepforest boxes                    
                    self.crowns = gpd.GeoDataFrame(pd.concat([self.crowns, IFAS]))
                
                self.crowns.to_file("{}/crowns.shp".format(self.data_dir))                
                if self.comet_logger:
                    self.comet_logger.experiment.log_parameter("Species after crown prediction", len(self.crowns.taxonID.unique()))
                    self.comet_logger.experiment.log_parameter("Samples after crown prediction", self.crowns.shape[0])
                
                if self.comet_logger:
                    self.comet_logger.experiment.log_parameter("Species after dead filtering",len(self.crowns.taxonID.unique()))
                    self.comet_logger.experiment.log_parameter("Samples after dead filtering",self.crowns.shape[0])
                    try:
                        for index, row in self.predicted_dead.iterrows():
                            left, bottom, right, top = row["geometry"].bounds                
                            img_path = neon_paths.find_sensor_path(lookup_pool=self.rgb_pool, bounds=row["geometry"].bounds)
                            src = rasterio.open(img_path)
                            img = src.read(window=rasterio.windows.from_bounds(left-4, bottom-4, right+4, top+4, transform=src.transform))                      
                            img = np.rollaxis(img, 0, 3)
                            self.comet_logger.experiment.log_image(image_data=img, name="Dead: {} ({:.2f}) {}".format(row["dead_label"],row["dead_score"],row["individual"]))                        
                    except:
                        print("No dead trees predicted")
            
            self.crowns = gpd.read_file("{}/crowns.shp".format(self.data_dir, site))
            if self.config["replace_crops"]:   
                
                #HSI crops
                self.annotations = generate.generate_crops(
                    self.crowns,
                    savedir=self.data_dir,
                    img_pool=self.hsi_pool,
                    h5_pool=self.h5_pool,
                    convert_h5=self.config["convert_h5"],   
                    rgb_pool=self.rgb_pool,
                    HSI_tif_dir=self.config["HSI_tif_dir"],
                    client=self.client,
                    as_numpy=True,
                    suffix="HSI"
                )
            
                self.annotations.to_csv("{}/HSI_annotations.csv".format(self.data_dir))
                
                rgb_crowns = self.crowns.copy()
                rgb_crowns.geometry = rgb_crowns.geometry.buffer(1)
                rgb_annotations = generate.generate_crops(
                    rgb_crowns,
                    savedir=self.data_dir,
                    img_pool=self.rgb_pool,
                    h5_pool=self.h5_pool,
                    rgb_pool=self.rgb_pool,
                    convert_h5=False,   
                    client=self.client,
                    suffix="RGB"
                )
                rgb_annotations["RGB_image_path"] = rgb_annotations["image_path"]

                rgb_annotations.to_csv("{}/RGB_annotations.csv".format(self.data_dir))

                self.annotations = self.annotations.merge(rgb_annotations[["individual","tile_year","RGB_image_path"]], on=["individual","tile_year"])
                
                self.annotations.to_csv("{}/annotations.csv".format(self.data_dir))
                
            else:
                self.annotations = pd.read_csv("{}/annotations.csv".format(self.data_dir))
            if self.comet_logger:
                self.comet_logger.experiment.log_parameter("Species after crop generation",len(self.annotations.taxonID.unique()))
                num_individuals = self.annotations.groupby("individual").apply(lambda x: x.head(1)).shape[0]
                self.comet_logger.experiment.log_parameter("Individuals after crop generation",num_individuals)                
                self.comet_logger.experiment.log_parameter("Samples after crop generation",self.annotations.shape[0])
    
            if create_train_test:
                self.train, self.test = self.create_train_test_split(self.experiment_id)  
            else:
                print("Loading a train-test split from {}/{}".format(self.data_dir, "{}_{}".format(self.config["train_test_commit"], site)))
                self.train = pd.read_csv("{}/test_{}.csv".format(self.data_dir, "{}_{}".format(self.config["train_test_commit"], site)))            
                self.test = pd.read_csv("{}/test_{}.csv".format(self.data_dir, "{}_{}".format(self.config["train_test_commit"], site)))               
        else:
            print("Loading previous data commit {}".format(self.config["use_data_commit"]))
            self.annotations = pd.read_csv("{}/annotations.csv".format(self.data_dir)) 
                
            if create_train_test:
                print("Using data commit {} creating a new train-test split for site {}".format(self.config["use_data_commit"],self.site))
                self.create_train_test_split(ID=self.experiment_id)
            else:
                print("Loading a train-test split from {}/{}".format(self.data_dir, "{}_{}".format(self.config["train_test_commit"], site)))
                self.train = pd.read_csv("{}/train_{}.csv".format(self.data_dir, "{}_{}".format(self.config["train_test_commit"], site)))            
                self.test = pd.read_csv("{}/test_{}.csv".format(self.data_dir, "{}_{}".format(self.config["train_test_commit"], site)))            
            
            self.crowns = gpd.read_file("{}/crowns.shp".format(self.data_dir))
                            
            #mimic schema due to abbreviation when .shp is saved
            self.crowns["individual"] = self.crowns["individual"]
            self.canopy_points = gpd.read_file("{}/canopy_points.shp".format(self.data_dir))
                
            self.canopy_points["individual"] = self.canopy_points["individual"]
        
        self.create_datasets(self.train, self.test)
        print("There are {} records for {} species for {} sites in filtered train".format(
            self.train.shape[0],
            len(self.train.label.unique()),
            len(self.train.siteID.unique())
        ))
        
        print("There are {} records for {} species for {} sites in test".format(
            self.test.shape[0],
            len(self.test.label.unique()),
            len(self.test.siteID.unique()))
        )    
    def create_train_test_split(self, ID):      
        if self.site:
            if "pretrain" not in self.site:
                # Get species present at site, as well as those species from other sites
                self.other_sites = self.annotations[~self.annotations.siteID.isin(self.site)].reset_index(drop=True)                
                self.annotations = self.annotations[self.annotations.siteID.isin(self.site)].reset_index(drop=True)
                self.other_sites = self.other_sites[self.other_sites.taxonID.isin(self.annotations.taxonID.unique())]
                
        if self.config["existing_test_csv"]:
            print("Reading in existing test_csv: {}".format(self.config["existing_test_csv"]))
            existing_test = pd.read_csv(self.config["existing_test_csv"])
            self.test = self.annotations[self.annotations.individual.isin(existing_test.individual)]  
            self.train = self.annotations[~self.annotations.individual.isin(existing_test.individual)]
            self.train = self.train[self.train.taxonID.isin(self.test.taxonID)].reset_index(drop=True)
        else:
            self.train, self.test = train_test_split(self.annotations, config=self.config, client=self.client) 

        # Capture discarded species
        if "pretrain" not in self.site:
            individuals = np.concatenate([self.train.individual.unique(), self.test.individual.unique()])
            self.novel = self.annotations[~self.annotations.individual.isin(individuals)]
            
            # Counts by discarded species
            keep = self.novel.groupby("individual").apply(lambda x: x.head(1)).taxonID.value_counts() > (self.config["min_test_samples"])
            species_to_keep = keep[keep].index
            self.novel = self.novel[self.novel.taxonID.isin(species_to_keep)]
            
            #Recover any individual from target site
            self.novel.to_csv("{}/novel_species_{}.csv".format(self.data_dir, self.site))  
            
        self.create_label_dict(self.train, self.test)

        #Encode the numeric class data
        self.train["label"] = self.train.taxonID.apply(lambda x: self.species_label_dict[x])            
        self.test["label"] = self.test.taxonID.apply(lambda x: self.species_label_dict[x])
        
        #make sure indexed to 0 after any filtering
        self.test = self.test.reset_index(drop=True)
        self.train = self.train.reset_index(drop=True)
        
        self.train.to_csv("{}/train_{}.csv".format(self.data_dir, ID), index=False)            
        self.test.to_csv("{}/test_{}.csv".format(self.data_dir, ID), index=False)            
        
        return self.train, self.test
    
    def create_datasets(self, train, test):
        #Store class labels
        self.create_label_dict(train, test)
        
        #Create dataloaders
        self.train_ds = TreeDataset(
            df = train,
            config=self.config,
        )
        
        self.val_ds = TreeDataset(
            df=test,
            config=self.config,
        ) 
    
    def create_label_dict(self, train, test):
        # Store class labels
        unique_species_labels = np.concatenate([train.taxonID.unique(), test.taxonID.unique()])
        unique_species_labels = np.unique(unique_species_labels)
        unique_species_labels = np.sort(unique_species_labels)            
        self.num_classes = len(unique_species_labels)

        # Taxon to ID dict and the reverse    
        self.species_label_dict = {}
        for index, taxonID in enumerate(unique_species_labels):
            self.species_label_dict[taxonID] = index            
        self.label_to_taxonID = {v: k  for k, v in self.species_label_dict.items()}        
        
    def train_dataloader(self):
        one_hot = torch.nn.functional.one_hot(torch.tensor(self.train.label.values))
        train_sampler = sampler.MultilabelBalancedRandomSampler(
            labels=one_hot, indices=self.train.index, class_choice="cycle")
                
        data_loader = torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.config["batch_size"],
            sampler=train_sampler,
            num_workers=self.config["workers"],
        )
        
        return data_loader
    
    def val_dataloader(self):
        data_loader = torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config["workers"],
        )
        
        return data_loader
        
