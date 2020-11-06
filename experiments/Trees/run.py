# Run Experiment
## Sleep for a moment to allow queries to build up in SLURM queue
import os
from random import randint
from time import sleep
from datetime import datetime

from comet_ml import Experiment
from DeepTreeAttention.trees import AttentionModel
from DeepTreeAttention.utils import metrics, resample, start_cluster
from DeepTreeAttention.models.layers import WeightedSum
from DeepTreeAttention.visualization import visualize

import tensorflow as tf
from tensorflow.keras import metrics as keras_metrics
from tensorflow.keras.models import load_model
from distributed import wait

import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def find_shapefiles(dirname):
    files = glob.glob(os.path.join(dirname,"*.shp"))
    return files

def predict(dirname, savedir, generate=True, cpus=2, parallel=True, height=40, width=40, channels=3):
    """Create a wrapper dask cluster and run list of shapefiles in parallel (optional)
        Args:
            dirname: directory of DeepForest predicted shapefiles to run
            savedir: directory to write processed shapefiles
            generate: Do tfrecords need to be generated/overwritten or use existing records?
            cpus: Number of dask cpus to run
    """
    shapefiles = find_shapefiles(dirname=dirname)
    
    if parallel:
        client = start_cluster.start(cpus=cpus)
        futures = client.map(_predict_,shapefiles, create_records=generate, savedir=savedir, height=height, width=width, channels=channels)
        wait(futures)
        
        for future in futures:
            print(future.result())
    else:
        for shapefile in shapefiles:
            _predict_(shapefile, model_path, savedir=savedir, create_records=generate)
            
if __name__ == "__main__":
    sleep(randint(0,20))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = "{}/{}".format("/orange/idtrees-collab/DeepTreeAttention/snapshots/",timestamp)
    os.mkdir(save_dir)
    
    experiment = Experiment(project_name="neontrees", workspace="bw4sz")
    experiment.add_tag("Train")

    #Create output folder
    experiment.log_parameter("timestamp",timestamp)
    
    #Create a class and run
    model = AttentionModel(config="/home/b.weinstein/DeepTreeAttention/conf/tree_config.yml")
    model.create()
        
    #Log config
    experiment.log_parameters(model.config["train"])
    experiment.log_parameters(model.config["evaluation"])    
    experiment.log_parameters(model.config["predict"])
    experiment.log_parameters(model.config["train"]["ensemble"])
    
    ##Train
    #Train see config.yml for tfrecords path with weighted classes in cross entropy
    model.read_data()
    
    #Log the size of the training data
    counter=0
    for data, label in model.train_split:
        counter += data.shape[0]
    experiment.log_parameter("Training Samples", counter)
        
    #Load from file and compile or train new models
    if model.config["train"]["checkpoint_dir"] is not None:
        dirname = model.config["train"]["checkpoint_dir"]        
        if model.config["train"]["gpus"] > 1:
            with model.strategy.scope():   
                print("Running in parallel on {} GPUs".format(model.strategy.num_replicas_in_sync))
                #model.RGB_model = load_model("{}/RGB_model.h5".format(dirname), custom_objects={"WeightedSum": WeightedSum}, compile=False)
                model.HSI_model = load_model("{}/HSI_model.h5".format(dirname), custom_objects={"WeightedSum": WeightedSum}, compile=False)  
                model.metadata_model = load_model("{}/metadata_model.h5".format(dirname), compile=False)  
        else:
            #model.RGB_model = load_model("{}/RGB_model.h5".format(dirname), custom_objects={"WeightedSum": WeightedSum})
            model.HSI_model = load_model("{}/HSI_model.h5".format(dirname), custom_objects={"WeightedSum": WeightedSum})     
            model.metadata_model = load_model("{}/metadata_model.h5".format(dirname), compile=False)  
                
    else:
        if model.config["train"]["pretrain"]:
            #metadata network
            with experiment.context_manager("metadata"):
                print("Train metadata")
                model.read_data(mode="metadata")
                model.train(submodel="metadata", experiment=experiment)
                model.metadata_model.save("{}/metadata_model.h5".format(save_dir))
                
            ###Train subnetworks
            #experiment.log_parameter("Train subnetworks", True)
            #with experiment.context_manager("RGB_spatial_subnetwork"):
                #print("Train RGB spatial subnetwork")
                #model.read_data(mode="RGB_submodel")
                #model.train(submodel="spatial", sensor="RGB", experiment=experiment)
                
            #with experiment.context_manager("RGB_spectral_subnetwork"):
                #print("Train RGB spectral subnetwork")    
                #model.read_data(mode="RGB_submodel")   
                #model.train(submodel="spectral", sensor="RGB", experiment=experiment)
                    
            ##Train full RGB model
            #with experiment.context_manager("RGB_model"):
                #experiment.log_parameter("Class Weighted", True)
                #model.read_data(mode="RGB_train")
                #model.train(sensor="RGB", experiment=experiment)
                #model.RGB_model.save("{}/RGB_model.h5".format(save_dir))
                
                ##Get Alpha score for the weighted spectral/spatial average. Higher alpha favors spatial network.
                #if model.config["train"]["RGB"]["weighted_sum"]:
                    #estimate_a = model.RGB_model.get_layer("weighted_sum").get_weights()
                    #experiment.log_metric(name="spatial-spectral weight", value=estimate_a[0][0])
            
            ##Train subnetwork
            experiment.log_parameter("Train subnetworks", True)
            with experiment.context_manager("HSI_spatial_subnetwork"):
                print("Train HSI spatial subnetwork")
                model.read_data(mode="HSI_submodel")
                model.train(submodel="spatial", sensor="hyperspectral", experiment=experiment)
            
            with experiment.context_manager("HSI_spectral_subnetwork"):
                print("Train HSI spectral subnetwork")    
                model.read_data(mode="HSI_submodel")   
                model.train(submodel="spectral", sensor="hyperspectral", experiment=experiment)
                    
            #Train full model
            with experiment.context_manager("HSI_model"):
                experiment.log_parameter("Class Weighted", True)
                model.read_data(mode="HSI_train")
                model.train(sensor="hyperspectral", experiment=experiment)
                model.HSI_model.save("{}/HSI_model.h5".format(save_dir))
                
                #Get Alpha score for the weighted spectral/spatial average. Higher alpha favors spatial network.
                if model.config["train"]["HSI"]["weighted_sum"]:
                    estimate_a = model.HSI_model.get_layer("weighted_sum").get_weights()
                    experiment.log_metric(name="spatial-spectral weight", value=estimate_a[0][0])
            
    ##Ensemble
    with experiment.context_manager("ensemble"):    
        print("Train Ensemble")
        model.ensemble(freeze=model.config["train"]["ensemble"]["freeze"], experiment=experiment)
    
    #Final score, be absolutely sure you get all the data, feed slowly in batches of 1
    final_score = model.ensemble_model.evaluate(model.val_split.unbatch().batch(1))    
    experiment.log_metric("Ensemble Accuracy", final_score[1])
    
    #Save model and figure
    #tf.keras.utils.plot_model(model.ensemble_model, to_file="{}/Ensemble.png".format(save_dir))
    #experiment.log_figure("{}/Ensemble.png".format(save_dir))
    model.ensemble_model.save("{}/Ensemble.h5".format(save_dir))
    
    #save predictions
    
