#Predict species class for a set of DeepForest boxes
from distributed import wait
import geopandas
from DeepTreeAttention.main import AttentionModel
from DeepTreeAttention.generators.boxes import generate_prediction
from DeepTreeAttention.utils import start_cluster
import glob
import os

def find_shapefiles(dirname):
    files = glob.glob(os.path.join(dirname,"*.shp"))
    return files

def find_hyperspectral_path(shapefile, lookup_dir = "/orange/ewhite/NeonData/"):
    """Find a hyperspec path based on the shapefile"""
    pool = glob.glob(lookup_dir + "*/DP3.30006.001/**/Reflectance/*.tif",recursive=True)
    basename = os.path.splitext(os.path.basename(shapefile))[0]
    match = [x for x in pool if basename in x]
    
    if len(match) == 0:
        raise ValueError("No matching tile in {} for shapefile {}".format(lookup_dir, shapefile))
    elif len(match) > 1:
        raise ValueError("Multiple matching tiles in {} for shapefile {}".format(lookup_dir, shapefile))
    else:
        return match

def generate_tfrecords(shapefile, model_path, savedir):
    """Predict species class for each DeepForest bounding box
    Args:
        shapefile: a DeepForest shapefile (see NeonCrownMaps) with a bounding box and utm projection
        model_path: Path to trained .h5 DeepTreeAttention model
    """
    hyperspectral_path = find_hyperspectral_path(shp)
    generate_prediction(shapefile = shapefile, sensor_path = hyperspectral_path, savedir=savedir)
    
def predict_tfrecords(dirname, saved_model):
    mod = AttentionModel(config="conf/config.yml", saved_model=saved_model)
    mod.create()
    
    #Predict each tfrecord and majority vote
    results = mod.predict_boxes(dirname, majority_vote=True)
    
    return results

def merge_shapefile(shapefile, results, savedir):
    """Merge predicted species label with box id"""
    
    gdf = geopandas.read_file(shapefile)
    basename = os.path.splitext(os.path.basename(shapefile))
    gdf["box_index"] = "{}_{}".format(basename, gdf["index"]) 
    joined_gdf = gdf.join(results, on="box_index")
    
    fname = "{}/{}.shp".format(savedir, basename)
    joined_gdf.to_file(fname)
    
def run(shapefile, model_path, savedir, record_dirname=".", generate=True):
    """Predict species id for each box in a single shapefile
    Args:
        shapefile: path to a shapefile
        record_dirname: directory to save generated records
    Returns:
        fname: path to predicted shapefile
    """
    if generate:
        generate_tfrecords(shapefile, model_path, savedir = record_dirname)
    
    results = predict_tfrecords(record_dirname, saved_model)
    
    #Merge with original box shapefile by index and write new shapefile to file
    merge_shapefile(shapefile, results, savedir=savedir )
    
def main(dirname, generate=True, cpus=2):
    """Create a dask cluster and run list of shapefiles in parallel
        Args:
            dirname: directory of DeepForest predicted shapefiles to run
            generate: Do tfrecords need to be generated/overwritten or use existing records?
            cpus: Number of dask cpus to run
            """
    shapefiles = find_shapefiles(dirname=dirname)
    client = start_cluster.start(cpus=cpus)
    futures = client.map(run,shapefiles)
    wait(futures)
    
    for future in futures:
        print(future.result())    
    
if __name == "__main__":
    main(dirname="/orange/idtrees-collab/predictions/", generate=True, cpus=2)
