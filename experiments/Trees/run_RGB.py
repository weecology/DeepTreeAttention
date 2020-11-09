#Linear metadata model for testing purposes
from comet_ml import Experiment
import tensorflow as tf
from DeepTreeAttention.trees import AttentionModel
from DeepTreeAttention.models import metadata
from DeepTreeAttention.callbacks import callbacks
import pandas as pd
from tensorflow.keras.models import load_model
from DeepTreeAttention.models.layers import WeightedSum

model = AttentionModel(config="/home/b.weinstein/DeepTreeAttention/conf/tree_config.yml")
model.create()

#Log config
experiment = Experiment(project_name="neontrees", workspace="bw4sz")
experiment.log_parameters(model.config["train"])
experiment.log_parameters(model.config["evaluation"])    
experiment.log_parameters(model.config["predict"])
experiment.add_tag("RGB")

##Train

#Train see config.yml for tfrecords path with weighted classes in cross entropy
##Train subnetwork
experiment.log_parameter("Train subnetworks", True)
with experiment.context_manager("RGB_spatial_subnetwork"):
    print("Train RGB spatial subnetwork")
    model.read_data(mode="RGB_submodel")
    model.train(submodel="spatial", sensor="RGB", experiment=experiment)

with experiment.context_manager("RGB_spectral_subnetwork"):
    print("Train RGB spectral subnetwork")    
    model.read_data(mode="RGB_submodel")   
    model.train(submodel="spectral", sensor="RGB", experiment=experiment)
        
#Train full model
with experiment.context_manager("RGB_model"):
    model.read_data(mode="RGB_train")
    model.train(sensor="RGB", experiment=experiment)
    
    #Get Alpha score for the weighted spectral/spatial average. Higher alpha favors spatial network.
    if model.config["train"]["RGB"]["weighted_sum"]:
        estimate_a = model.RGB_model.get_layer("weighted_sum").get_weights()
        experiment.log_metric(name="spatial-spectral weight", value=estimate_a[0][0])
        
#Load RGB model
model.HSI_model = load_model("{}/HSI_model.h5".format(model.config["train"]["checkpoint_dir"]), custom_objects={"WeightedSum": WeightedSum})     
model.metadata_model = load_model("{}/metadata_model.h5".format(model.config["train"]["checkpoint_dir"]), compile=False)  
model.RGB_model = model.RGB_spatial

with experiment.context_manager("ensemble"):    
    print("Train Ensemble")
    model.ensemble(freeze=model.config["train"]["ensemble"]["freeze"], experiment=experiment)

#Final score, be absolutely sure you get all the data, feed slowly in batches of 1
final_score = model.ensemble_model.evaluate(model.val_split.unbatch().batch(1))    
experiment.log_metric("Ensemble Accuracy", final_score[1])