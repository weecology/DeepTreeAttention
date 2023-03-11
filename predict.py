from src import predict
from src import data
from src import neon_paths
from glob import glob
import geopandas as gpd

import traceback
from src.start_cluster import start
from src.models import multi_stage
from distributed import wait
import os
import re
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning import Trainer

def find_rgb_files(site, config, year="2021"):
    tiles = glob(config["rgb_sensor_pool"], recursive=True)
    tiles = [x for x in tiles if site in x]
    tiles = [x for x in tiles if "neon-aop-products" not in x]
    tiles = [x for x in tiles if "/{}/".format(year) in x]
    
    return tiles


def convert(rgb_path, hyperspectral_pool, savedir):
    #convert .h5 hyperspec tile if needed
    basename = os.path.basename(rgb_path)
    geo_index = re.search("(\d+_\d+)_image", basename).group(1)
    
    h5_list = [x for x in hyperspectral_pool if geo_index in x]
    tif_paths = []
    for path in h5_list:
        year = path.split("/")[6]
        tif_basename = os.path.splitext(os.path.basename(rgb_path))[0] + "_hyperspectral_{}.tif".format(year)
        tif_path = "{}/{}".format(savedir, tif_basename)
        if not os.path.exists(tif_path):
            tif_paths.append(neon_paths.convert_h5(path, rgb_path, savedir, year=year))
        else:
            tif_paths.append(tif_path)
    
    return tif_paths

#Params
config = data.read_config("config.yml")
config["preload_images"] = False
comet_logger = CometLogger(project_name="DeepTreeAttention2", workspace=config["comet_workspace"], auto_output_logging="simple")    
comet_logger.experiment.add_tag("prediction")

comet_logger.experiment.log_parameters(config)

cpu_client = start(cpus=20, mem_size="10GB")

dead_model_path = "/orange/idtrees-collab/DeepTreeAttention/Dead/snapshots/c4945ae57f4145948531a0059ebd023c.pl"
config["crop_dir"] = "/blue/ewhite/b.weinstein/DeepTreeAttention/3fd4871aede4484b9a6d20817d520185"
savedir = config["crop_dir"] 

species_model_paths = {"TEAK":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/d14321cbbaec4559b070ddb1b57f7de1_['TEAK', 'SOAP', 'YELL', 'ABBY'].pt",
                       "SOAP":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/d14321cbbaec4559b070ddb1b57f7de1_['TEAK', 'SOAP', 'YELL', 'ABBY'].pt"}                    

def create_landscape_map(site, model_path, config, cpu_client):
    #generate HSI_tif data if needed.
    h5_pool = glob(config["HSI_sensor_pool"], recursive=True)
    h5_pool = [x for x in h5_pool if not "neon-aop-products" in x]
    
    ### Step 1 Find RGB Tiles and convert HSI
    tiles = find_rgb_files(site=site, config=config, year="2020")[:5]
    tif_futures = cpu_client.map(
        convert,
        tiles,
        hyperspectral_pool=h5_pool,
        savedir=config["HSI_tif_dir"])
    wait(tif_futures)
    
    for x in tiles[:2]:
        basename = os.path.splitext(os.path.basename(x))[0]                
        shpname = "/blue/ewhite/b.weinstein/DeepTreeAttention/results/crowns/{}.shp".format(basename)      
        if not os.path.exists(shpname):
            try:
                crowns = predict.find_crowns(rgb_path=x, config=config, dead_model_path=dead_model_path)   
                crowns.to_file(shpname)            
            except Exception as e:
                traceback.print_exc()
                print("{} failed to build crowns with {}".format(shpname, e))
                continue
    
    crown_annotations_paths = []
    crown_annotations_futures = []
    for x in tiles:
        basename = os.path.splitext(os.path.basename(x))[0]                
        shpname = "/blue/ewhite/b.weinstein/DeepTreeAttention/results/crowns/{}.shp".format(basename)    
        try:
            crowns = gpd.read_file(shpname)    
        except:
            continue
        if not os.path.exists("/blue/ewhite/b.weinstein/DeepTreeAttention/results/crops/{}.shp".format(basename)):
            written_file = predict.generate_prediction_crops(crowns, config, as_numpy=True, client=cpu_client)
            crown_annotations_paths.append(written_file)
        else:
            crown_annotations_path = "/blue/ewhite/b.weinstein/DeepTreeAttention/results/crops/{}.shp".format(basename)       
            crown_annotations_paths.append(crown_annotations_path)
            
    #Recursive predict to avoid prediction levels that will be later ignored.
    trainer = Trainer(gpus=config["gpus"], logger=False, enable_checkpointing=False)
    
    ## Step 2 - Predict Crowns
    print(model_path)
    # Load species model
    #Do not preload weights
    config["pretrained_state_dict"] = None
    m = multi_stage.MultiStage.load_from_checkpoint(model_path, config=config, train_mode=False)
    prediction_dir = os.path.join("/blue/ewhite/b.weinstein/DeepTreeAttention/results/",
                                  os.path.splitext(os.path.basename(model_path))[0])    
    try:
        os.mkdir(prediction_dir)
    except:
        pass
    for x in crown_annotations_paths:
        results_shp = os.path.join(prediction_dir, os.path.basename(x))  
        if not os.path.exists(results_shp):  
            print(x)
            try:
                predict.predict_tile(
                        crown_annotations=x,
                        filter_dead=True,
                        trainer=trainer,
                        m=m,
                        savedir=prediction_dir,
                        config=config)
            except Exception as e:
                traceback.print_exc()
                continue

for site, model_path in species_model_paths.items():
    create_landscape_map(site, model_path, config, cpu_client)
