#OSBS mining
from src import predict
from src import data
from glob import glob

def find_files(site, config):
    tiles = glob(config["HSI_tif_dir"])
    tiles = [x for x in tiles if site in x]
    
    return tiles

config = data.read_config("config.yml")
model_path = "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/additional_oasis_1838.pl"

predictions = []
tiles = find_files(site="OSBS", config=config)
for x in tiles:
    trees = predict.predict_tile(PATH=x, model_path=model_path, config=config, min_score=0.7, taxonIDs=["PICL","MAGNO","CAGL8"])
    if not trees.empty:
        predictions.append(trees)

predictions.to_file("results/OSBS_predictions.shp")