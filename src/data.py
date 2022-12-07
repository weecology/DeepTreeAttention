#Ligthning data module
from . import __file__
from distributed import wait
import glob
import geopandas as gpd
import numpy as np
import os
import pandas as pd
from pytorch_lightning import LightningDataModule
from src import generate
from src import CHM
from src import augmentation
from src import megaplot
from src import neon_paths
from src.models import dead
from src.utils import *

from shapely.geometry import Point
import torch
from torch.utils.data import Dataset
import rasterio

def filter_data(path, config):
    """Transform raw NEON data into clean shapefile   
    Args:
        config: DeepTreeAttention config dict, see config.yml
    """
    field = pd.read_csv(path)
    field["individual"] = field["individualID"]
    field = field[~field.itcEasting.isnull()]
    field = field[~field.growthForm.isin(["liana","small shrub"])]
    field = field[~field.growthForm.isnull()]
    field = field[~field.plantStatus.isnull()]        
    field = field[field.plantStatus.str.contains("Live")]    
    
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
    field.loc[field.taxonID=="PSMEM","taxonID"] = "PSME"
    field.loc[field.taxonID=="BEPAP","taxonID"] = "BEPA"
    field.loc[field.taxonID=="ACNEN","taxonID"] = "ACNE2"
    field.loc[field.taxonID=="ACRUR","taxonID"] = "ACRU"
    field.loc[field.taxonID=="PICOL","taxonID"] = "PICO"
    field.loc[field.taxonID=="ABLAL","taxonID"] = "ABLA"
    field.loc[field.taxonID=="ACSA3","taxonID"] = "ACSAS"
    field.loc[field.taxonID=="CECAC","taxonID"] = "CECA4"
    field.loc[field.taxonID=="PRSES","taxonID"] = "PRSE2"
    field.loc[field.taxonID=="PIPOS","taxonID"] = "PIPO"
    field.loc[field.taxonID=="BEPAC2","taxonID"] = "BEPA"
    field.loc[field.taxonID=="JUVIV","taxonID"] = "JUVI"
    field.loc[field.taxonID=="PRPEP","taxonID"] = "PRPE2"
    field.loc[field.taxonID=="COCOC","taxonID"] = "COCO6"
    field.loc[field.taxonID=="NYBI","taxonID"] = "NYSY"
    
    field = field[~field.taxonID.isin(["BETUL", "FRAXI", "HALES", "PICEA", "PINUS", "QUERC", "ULMUS", "2PLANT"])]
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
    keep = shp.taxonID.value_counts() > (min_sampled)
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
        saved_train, saved_test = ties[np.argmax([x[1].shape[0] for x in ties])]
        
    train = saved_train
    test = saved_test    
    
    # Give tests a unique index to match against
    test["point_id"] = test.index.values
    train["point_id"] = train.index.values
    
    return train, test

# Dataset class
class TreeDataset(Dataset):
    """A csv file with a path to image crop and label
    Args:
       csv_file: path to csv file with image_path and label
    """
    def __init__(self, csv_file, config=None, train=True, HSI=True, metadata=False):
        self.annotations = pd.read_csv(csv_file)
        self.train = train
        self.HSI = HSI
        self.metadata = metadata
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
        image_path = self.annotations.image_path.loc[index]      
        individual = os.path.basename(image_path.split(".tif")[0])
        if self.HSI:
            if self.config["preload_images"]:
                inputs["HSI"] = self.image_dict[index]
            else:
                image_basename = self.annotations.image_path.loc[index]  
                image_path = os.path.join(self.config["crop_dir"],image_basename)                
                image = load_image(image_path, image_size=self.image_size)
                inputs["HSI"] = image

        if self.metadata:
            site = self.annotations.site.loc[index]  
            site = torch.tensor(site, dtype=torch.int)
            inputs["site"] = site

        if self.train:
            label = self.annotations.label.loc[index]
            label = torch.tensor(label, dtype=torch.long)

            if self.HSI:
                inputs["HSI"] = self.transformer(inputs["HSI"])

            return individual, inputs, label
        else:
            return individual, inputs

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
    def __init__(self, csv_file, config, HSI=True, metadata=False, client = None, data_dir=None, comet_logger=None, debug=False):
        """
        Args:
            config: optional config file to override
            data_dir: override data location, defaults to ROOT   
            regenerate: Whether to recreate raw data
            debug: a test mode for small samples
        """
        super().__init__()
        self.ROOT = os.path.dirname(os.path.dirname(__file__))
        self.csv_file = csv_file
        self.HSI = HSI
        self.metadata = metadata
        self.comet_logger = comet_logger
        self.debug = debug 

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
            if self.config["replace"]: 
                    
                # Convert raw neon data to x,y tree locatins
                df = filter_data(self.csv_file, config=self.config)
                df = df[df.siteID.isin(["OSBS","TALL","JERC"])]
                
                # Load any megaplot data
                if not self.config["megaplot_dir"] is None:
                    megaplot_data = megaplot.load(directory=self.config["megaplot_dir"], config=self.config, client=self.client, site="OSBS")
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
                    
                df.to_file("{}/unfiltered_points.shp".format(self.data_dir))
                
                #Filter points based on LiDAR height for NEON data
                df = CHM.filter_CHM(df, CHM_pool=self.config["CHM_pool"],
                                    min_CHM_height=self.config["min_CHM_height"], 
                                    max_CHM_diff=self.config["max_CHM_diff"], 
                                    CHM_height_limit=self.config["CHM_height_limit"])  
                
                self.canopy_points = df
                self.canopy_points.to_file("{}/canopy_points.shp".format(self.data_dir))

                if self.comet_logger:
                    self.comet_logger.experiment.log_parameter("Species after CHM filter", len(df.taxonID.unique()))
                    self.comet_logger.experiment.log_parameter("Samples after CHM filter", df.shape[0])
            
                # Create crown data
                self.crowns = generate.points_to_crowns(
                    field_data="{}/canopy_points.shp".format(self.data_dir),
                    rgb_dir=self.config["rgb_sensor_pool"],
                    savedir="{}/boxes/".format(self.data_dir),
                    raw_box_savedir="{}/boxes/".format(self.data_dir)
                )
                
                if self.config["megaplot_dir"]:
                    #ADD IFAS back in, use polygons instead of deepforest boxes                    
                    self.crowns = gpd.GeoDataFrame(pd.concat([self.crowns, IFAS]))
                
                if self.comet_logger:
                    self.crowns.to_file("{}/crowns.shp".format(self.data_dir))
                    self.comet_logger.experiment.log_parameter("Species after crown prediction", len(self.crowns.taxonID.unique()))
                    self.comet_logger.experiment.log_parameter("Samples after crown prediction", self.crowns.shape[0])
                
                if self.comet_logger:
                    self.comet_logger.experiment.log_parameter("Species after dead filtering",len(self.crowns.taxonID.unique()))
                    self.comet_logger.experiment.log_parameter("Samples after dead filtering",self.crowns.shape[0])
                    try:
                        rgb_pool = glob.glob(self.config["rgb_sensor_pool"], recursive=True)
                        for index, row in self.predicted_dead.iterrows():
                            left, bottom, right, top = row["geometry"].bounds                
                            img_path = neon_paths.find_sensor_path(lookup_pool=rgb_pool, bounds=row["geometry"].bounds)
                            src = rasterio.open(img_path)
                            img = src.read(window=rasterio.windows.from_bounds(left-4, bottom-4, right+4, top+4, transform=src.transform))                      
                            img = np.rollaxis(img, 0, 3)
                            self.comet_logger.experiment.log_image(image_data=img, name="Dead: {} ({:.2f}) {}".format(row["dead_label"],row["dead_score"],row["individual"]))                        
                    except:
                        print("No dead trees predicted")
            else:
                self.crowns = gpd.read_file("{}/crowns.shp".format(self.data_dir))
                    
            annotations = generate.generate_crops(
                self.crowns,
                savedir=self.data_dir,
                sensor_glob=self.config["HSI_sensor_pool"],
                convert_h5=self.config["convert_h5"],   
                rgb_glob=self.config["rgb_sensor_pool"],
                HSI_tif_dir=self.config["HSI_tif_dir"],
                client=self.client,
                replace=self.config["replace"],
                as_numpy=True
            )
            
            #hard sampling cutoff
            #annotations = annotations.groupby("taxonID").apply(lambda x: x.head(self.config["sampling_ceiling"])).reset_index(drop=True)
            annotations.to_csv("{}/annotations.csv".format(self.data_dir))
        
            if self.comet_logger:
                self.comet_logger.experiment.log_parameter("Species after crop generation",len(annotations.taxonID.unique()))
                self.comet_logger.experiment.log_parameter("Samples after crop generation",annotations.shape[0])
            
            #Remove crowns from test dataset if specified
            if self.config["existing_test_csv"]:
                existing_test = pd.read_csv(self.config["existing_test_csv"])
                self.test = annotations[annotations.individual.isin(existing_test.individual)]  
                self.train = annotations[~annotations.individual.isin(existing_test.individual)]
                self.train = self.train[self.train.taxonID.isin(self.test.taxonID)]
                self.train = self.train.groupby("taxonID").apply(lambda x: x.head(self.config["sampling_ceiling"])).reset_index(drop=True)
            else:
                self.train, self.test = train_test_split(annotations, config=self.config, client=self.client) 

            # Capture discarded species
            individuals = np.concatenate([self.train.individual.unique(), self.test.individual.unique()])
            self.novel = annotations[~annotations.individual.isin(individuals)]
            self.novel = self.novel[~self.novel.taxonID.isin(np.concatenate([self.train.taxonID.unique(), self.test.taxonID.unique()]))]
            self.novel.to_csv("{}/novel_species.csv".format(self.data_dir))
            
            # Store class labels
            unique_species_labels = np.concatenate([self.train.taxonID.unique(), self.test.taxonID.unique()])
            unique_species_labels = np.unique(unique_species_labels)
            unique_species_labels = np.sort(unique_species_labels)            
            self.num_classes = len(unique_species_labels)
    
            # Taxon to ID dict and the reverse    
            self.species_label_dict = {}
            for index, taxonID in enumerate(unique_species_labels):
                self.species_label_dict[taxonID] = index            
            self.label_to_taxonID = {v: k  for k, v in self.species_label_dict.items()}
            
            #Encode the numeric site and class data
            self.train["label"] = self.train.taxonID.apply(lambda x: self.species_label_dict[x])            
            self.test["label"] = self.test.taxonID.apply(lambda x: self.species_label_dict[x])
            
            self.train.to_csv("{}/train.csv".format(self.data_dir), index=False)            
            self.test.to_csv("{}/test.csv".format(self.data_dir), index=False)
            
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
             
            #Create dataloaders
            self.train_ds = TreeDataset(
                csv_file = "{}/train.csv".format(self.data_dir),
                config=self.config,
            )
            
            self.val_ds = TreeDataset(
                csv_file = "{}/test.csv".format(self.data_dir),
                config=self.config,
            )
             
        else:
            print("Loading previous run")            
            self.train = pd.read_csv("{}/train.csv".format(self.data_dir))
            self.test = pd.read_csv("{}/test.csv".format(self.data_dir))
            
            try:
                self.train["individual"] = self.train["individualID"]
                self.test["individual"] = self.test["individualID"]
            except:
                pass
            
            self.crowns = gpd.read_file("{}/crowns.shp".format(self.data_dir))
            
            #mimic schema due to abbreviation when .shp is saved
            self.crowns["individual"] = self.crowns["individual"]
            self.canopy_points = gpd.read_file("{}/canopy_points.shp".format(self.data_dir))
            self.canopy_points["individual"] = self.canopy_points["individual"]
            
            #Store class labels
            unique_species_labels = np.concatenate([self.train.taxonID.unique(), self.test.taxonID.unique()])
            unique_species_labels = np.unique(unique_species_labels)
            unique_species_labels = np.sort(unique_species_labels)            
            self.num_classes = len(unique_species_labels)
            
            #Taxon to ID dict and the reverse    
            self.species_label_dict = {}
            for index, taxonID in enumerate(unique_species_labels):
                self.species_label_dict[taxonID] = index
