#Create benchmark
import geopandas as gpd
import pandas as pd
from src import generate
from src import data
from src import start_cluster
import os
import shutil

train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")

config = data.read_config("config.yml")
gdf = gpd.read_file("data/processed/crowns.shp")
points = gpd.read_file("data/processed/canopy_points.shp")
gdf = gdf[~gdf.individual.str.contains("contrib")]
gdf = gdf[gdf.siteID=="OSBS"]
gdf["RGB_tile"] = None

#HSI crops
#client = start_cluster.start(cpus=100)
client = None
annotations = generate.generate_crops(gdf.head(), sensor_glob=config["HSI_sensor_pool"], savedir="/blue/ewhite/b.weinstein/species_benchmark/HSI/", rgb_glob=config["rgb_sensor_pool"], client=client, convert_h5=True, HSI_tif_dir=config["HSI_tif_dir"])
generate.generate_crops(gdf.head(), sensor_glob=config["rgb_sensor_pool"], savedir="/blue/ewhite/b.weinstein/species_benchmark/RGB/", rgb_glob=config["rgb_sensor_pool"], client=client)
generate.generate_crops(gdf.head(), sensor_glob=config["CHM_pool"], savedir="/blue/ewhite/b.weinstein/species_benchmark/CHM/", rgb_glob=config["rgb_sensor_pool"], client=client)

points = points[["taxonID","site","individual","siteID","eventID","stemDiamet","plantStatu","elevation","canopyPosi","utmZone","itcEasting","itcNorthin","CHM_height","geometry"]]
points.label = points.taxonID.astype("category").cat.codes
train_annotations = annotations[annotations.individualID.isin(train.individualID)]
test_annotations = annotations[annotations.individualID.isin(test.individualID)]

for i in train_annotations.individualID:
    try:
        os.mkdir("/blue/ewhite/b.weinstein/species_benchmark/zenodo/train/{}".format(i))
    except:
        pass
    shutil.copy("/blue/ewhite/b.weinstein/species_benchmark/CHM/{}.tif".format(i),"/blue/ewhite/b.weinstein/species_benchmark/zenodo/train/{}/{}_CHM.tif".format(i,i))
    shutil.copy("/blue/ewhite/b.weinstein/species_benchmark/RGB/{}.tif".format(i),"/blue/ewhite/b.weinstein/species_benchmark/zenodo/train/{}/{}_RGB.tif".format(i,i))
    shutil.copy("/blue/ewhite/b.weinstein/species_benchmark/HSI/{}.tif".format(i),"/blue/ewhite/b.weinstein/species_benchmark/zenodo/train/{}/{}_HSI.tif".format(i,i))

train_points = points[points.individual.isin(train_annotations.individualID)]
points.to_file("/blue/ewhite/b.weinstein/species_benchmark/zenodo/train/label.shp")

for i in test_annotations.individualID:
    try:
        os.mkdir("/blue/ewhite/b.weinstein/species_benchmark/zenodo/test/{}".format(i))
    except:
        pass
    shutil.copy("/blue/ewhite/b.weinstein/species_benchmark/CHM/{}.tif".format(i),"/blue/ewhite/b.weinstein/species_benchmark/zenodo/test/{}/{}_CHM.tif".format(i,i))
    shutil.copy("/blue/ewhite/b.weinstein/species_benchmark/RGB/{}.tif".format(i),"/blue/ewhite/b.weinstein/species_benchmark/zenodo/test/{}/{}_RGB.tif".format(i,i))
    shutil.copy("/blue/ewhite/b.weinstein/species_benchmark/HSI/{}.tif".format(i),"/blue/ewhite/b.weinstein/species_benchmark/zenodo/test/{}/{}_HSI.tif".format(i,i))

test_points = points[points.individual.isin(annotations.individualID)]
test_points.to_file("/blue/ewhite/b.weinstein/species_benchmark/zenodo/test/label.shp")
