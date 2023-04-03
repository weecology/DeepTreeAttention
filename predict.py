import geopandas as gpd
import traceback
from src.start_cluster import start
from distributed import wait, as_completed, fire_and_forget
import os
import re
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning import Trainer

from src import predict
from src import data
from src import neon_paths
from src.utils import create_glob_lists 

def find_rgb_files(site, rgb_pool, year="2020"):
    tiles = [x for x in rgb_pool if site in x]
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

client = start(cpus=30, mem_size="8GB")

dead_model_path = "/orange/idtrees-collab/DeepTreeAttention/Dead/snapshots/c4945ae57f4145948531a0059ebd023c.pl"
config["crop_dir"] = "/blue/ewhite/b.weinstein/DeepTreeAttention/results/site_crops"
savedir = config["crop_dir"] 

species_model_paths = {
    #"NIWO": "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/234a38632f2142beb82e7d0ad701e4f7_['NIWO', 'RMNP'].pt",
    #"SJER":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/0c580b6730614574bc232245422a2600_['SJER'].pt",
    #"MOAB":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/7c7abe8e0b0040e1b7d7a62c1d8926e5_['MOAB', 'REDB'].pt",
    #"WREF":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/ab9eee4c0c6b44ce9a6fda25beab8e83_['WREF'].pt",
    #"REDB":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/7c7abe8e0b0040e1b7d7a62c1d8926e5_['MOAB', 'REDB'].pt",
    "SERC":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/cd1739965ac54da781b9cbc89ed4f131_['SERC', 'GRSM'].pt",
    "GRSM":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/cd1739965ac54da781b9cbc89ed4f131_['SERC', 'GRSM'].pt",
    "DEJU":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/da0d5a8500e54f0599bce7876e397f89_['BONA', 'DEJU'].pt",
    "BONA":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/da0d5a8500e54f0599bce7876e397f89_['BONA', 'DEJU'].pt",
    "TREE":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/0b8ea07340e44ba79f29682af1e93f3b_['TREE', 'STEI', 'UNDE'].pt",
    "STEI":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/0b8ea07340e44ba79f29682af1e93f3b_['TREE', 'STEI', 'UNDE'].pt",
    "UNDE":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/0b8ea07340e44ba79f29682af1e93f3b_['TREE', 'STEI', 'UNDE'].pt",
    "DELA":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/9865034de3cc4ec4b861c32e1bae19b7_['DELA', 'LENO'].pt",
    "LENO":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/9865034de3cc4ec4b861c32e1bae19b7_['DELA', 'LENO'].pt",
    "OSBS":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/db9049a717634f049636ec7fd4c66b7a_['OSBS', 'JERC', 'TALL', 'DSNY'].pt",
    "JERC":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/db9049a717634f049636ec7fd4c66b7a_['OSBS', 'JERC', 'TALL', 'DSNY'].pt",
    "TALL":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/db9049a717634f049636ec7fd4c66b7a_['OSBS', 'JERC', 'TALL', 'DSNY'].pt",
    "DSNY":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/db9049a717634f049636ec7fd4c66b7a_['OSBS', 'JERC', 'TALL', 'DSNY'].pt",
    "CLBJ":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/c51bfd7418ba4e198bdd31bc852e6d44_['CLBJ', 'KONZ'].pt",
    "TEAK":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/6c292d5a990f420e81fe69f5697457ef_['TEAK', 'SOAP', 'YELL', 'ABBY'].pt",
    "SOAP":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/6c292d5a990f420e81fe69f5697457ef_['TEAK', 'SOAP', 'YELL', 'ABBY'].pt",
    "YELL":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/6c292d5a990f420e81fe69f5697457ef_['TEAK', 'SOAP', 'YELL', 'ABBY'].pt",                       
    "MLBS":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/6c292d5a990f420e81fe69f5697457ef_['MLBS','BLAN','SCBI','UKFS'].pt",
    "BLAN":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/6c292d5a990f420e81fe69f5697457ef_['MLBS','BLAN','SCBI','UKFS'].pt",
    "SCBI":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/6c292d5a990f420e81fe69f5697457ef_['MLBS','BLAN','SCBI','UKFS'].pt",
    "UKFS":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/6c292d5a990f420e81fe69f5697457ef_['MLBS','BLAN','SCBI','UKFS'].pt",
    "BART":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/15d88bbd39ea43faaa3abd0867ef5dee_['BART', 'HARV'].pt",
    "HARV":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/15d88bbd39ea43faaa3abd0867ef5dee_['BART', 'HARV'].pt"}

def create_landscape_map(site, model_path, config, client, rgb_pool, hsi_pool, h5_pool, CHM_pool):
    #Prepare directories
    # Crop Predicted Crowns
    try:
        prediction_dir = os.path.join("/blue/ewhite/b.weinstein/DeepTreeAttention/results/",
                                      os.path.splitext(os.path.basename(model_path))[0])     
        os.mkdir(prediction_dir)        
    except:
        pass
    
    try:
        os.mkdir("/blue/ewhite/b.weinstein/DeepTreeAttention/results/site_crops/{}".format(site))
    except:
        pass

    ### Step 1 Find RGB Tiles and convert HSI, prioritize 2022
    for year in [2022, 2021, 2020, 2019]:
        tiles = find_rgb_files(site=site, rgb_pool=rgb_pool, year=year)
        if len(tiles) > 0:
            break
        
    if len(tiles) == 0:
        raise ValueError("There are no RGB tiles for any year since 2019 for {}".format(site))
    
    tif_futures = client.map(
        convert,
        tiles,
        hyperspectral_pool=h5_pool,
        savedir=config["HSI_tif_dir"]
    )
    wait(tif_futures)
    
    species_futures = []
    crop_futures = []
    
    # Predict crowns
    for x in tiles:
        print(x)
        try:
            crown_path = predict.find_crowns(
                rgb_path=x,
                config=config,
                dead_model_path=dead_model_path,
                savedir="/blue/ewhite/b.weinstein/DeepTreeAttention/results/crowns",
                overwrite=False)
        except:
            traceback.print_exc()

        crop_future = client.submit(
            predict.generate_prediction_crops,
            crown_path,
            config,
            crop_dir="/blue/ewhite/b.weinstein/DeepTreeAttention/results/site_crops/{}".format(site),
            as_numpy=True,
            client=None,
            img_pool=hsi_pool,
            h5_pool=h5_pool,
            rgb_pool=rgb_pool,
            overwrite=False
        )
        fire_and_forget(crop_future)
        crop_futures.append(crop_future)
        
    return crop_futures
        #if not os.path.exists(results_shp):  
            #species_future = gpu_client.submit(
                #predict.predict_tile, 
                #crown_annotations=crown_annotations_path,
                #filter_dead=True,
                #model_path=model_path,
                #savedir=prediction_dir,
                #config=config)
            #species_futures.append(species_future)
    #for x in species_futures:
        #try:
            #x.result()
        #except:
            #traceback.print_exc()
            #continue

        
#generate HSI_tif data if needed.
rgb_pool, h5_pool, hsi_pool, CHM_pool = create_glob_lists(config)
for site, model_path in species_model_paths.items():
    print(site)
    try:
        futures = create_landscape_map(
            site,
            model_path,
            config,
            client, 
            rgb_pool=rgb_pool,
            h5_pool=h5_pool,
            hsi_pool=hsi_pool,
            CHM_pool=CHM_pool
        )
    except:
        traceback.print_exc()
        continue
    
wait(crop_futures)
