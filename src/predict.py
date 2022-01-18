#Predict
from deepforest import main
from deepforest.utilities import annotations_to_shapefile
import glob
import rasterio
from src.main import TreeModel
from src import data 
from torch.utils.data import Dataset
import os
from torchvision import transforms
from torch.nn import functional as F
import torch
from torch.utils.data.dataloader import default_collate

class on_the_fly_Dataset(Dataset):
    """A csv file with a path to image crop and label
    Args:
       crowns: geodataframe of crown locations from a single rasterio src
       image_path: .tif file location
    """
    def __init__(self, crowns, image_path, config=None):
        self.config = config 
        self.crowns = crowns
        self.image_size = config["image_size"]
        self.src = rasterio.open(image_path)
        
    def __len__(self):
        #0th based index
        return self.crowns.shape[0]
        
    def __getitem__(self, index):
        geom = self.crowns.iloc[index].geometry
        individual = self.crowns.iloc[index].individual
        left, bottom, right, top = geom.bounds
        crop = self.src.read(window=rasterio.windows.from_bounds(left, bottom, right, top, transform=self.src.transform)) 
        
        if crop.size == 0:
            return individual, None
            
        #preprocess and batch
        image = data.preprocess_image(crop, channel_is_first=True)
        image = transforms.functional.resize(image, size=(self.config["image_size"],self.config["image_size"]), interpolation=transforms.InterpolationMode.NEAREST)
        
        inputs = {}
        inputs["HSI"] = image
    
        return individual, inputs
        

def my_collate(batch):
    batch = [x for x in batch if x[1] is not None]
    return default_collate(batch)
    
def predict_tile(PATH, model_path, config, min_score, taxonIDs):
    #get rgb from HSI path
    HSI_basename = os.path.basename(PATH)
    if "hyperspectral" in HSI_basename:
        rgb_name = "{}.tif".format(HSI_basename.split("_hyperspectral")[0])    
    else:
        rgb_name = HSI_basename           
    rgb_pool = glob.glob(config["rgb_sensor_pool"], recursive=True)
    rgb_path = [x for x in rgb_pool if rgb_name in x][0]
    crowns = predict_crowns(rgb_path)
    crowns["tile"] = PATH
    trees = predict_species(HSI_path=PATH, crowns=crowns, model_path=model_path, config=config)
    chosen_trees = choose_trees(trees, min_score=min_score, taxonIDs=taxonIDs)
    
    return chosen_trees

def predict_crowns(PATH):
    m = main.deepforest()
    if torch.cuda.is_available():
        m.config["gpus"] = 1
    m.use_release(check_release=False)
    boxes = m.predict_tile(PATH)
    r = rasterio.open(PATH)
    transform = r.transform     
    crs = r.crs
    gdf = annotations_to_shapefile(boxes, transform=transform, crs=crs)
    
    #Dummy variables for schema
    basename = os.path.splitext(os.path.basename(PATH))[0]
    individual = ["{}_{}".format(x, basename) for x in range(gdf.shape[0])]
    gdf["individual"] = individual
    gdf["plotID"] = None
    gdf["siteID"] = None #TODO
    gdf["box_id"] = None
    gdf["plotID"] = None
    gdf["taxonID"] = None
    
    return gdf

def predict_species(crowns, HSI_path, model_path, config):
    m = TreeModel.load_from_checkpoint(model_path)
    ds = on_the_fly_Dataset(crowns, HSI_path, config)
    data_loader = torch.utils.data.DataLoader(
        ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["workers"],
        collate_fn=my_collate
    )
    df = m.predict_dataloader(data_loader, train=False)
    df = df.merge(crowns[["individual","geometry"]], on="individual")
    
    return df

def choose_trees(trees, min_score, taxonIDs):
    trees = trees[trees.top1_score > min_score]
    trees = trees[trees.pred_taxa_top1.isin(taxonIDs)]
    
    return trees