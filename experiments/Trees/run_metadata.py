#Linear metadata model for testing purposes
from comet_ml import Experiment
import tensorflow as tf
from DeepTreeAttention.trees import AttentionModel
from DeepTreeAttention.models import metadata
from DeepTreeAttention.callbacks import callbacks
import pandas as pd

model = AttentionModel(config="/home/b.weinstein/DeepTreeAttention/conf/tree_config.yml")
model.create()

#Log config
experiment = Experiment(project_name="neontrees", workspace="bw4sz")
experiment.log_parameters(model.config["train"])
experiment.log_parameters(model.config["evaluation"])    
experiment.log_parameters(model.config["predict"])
experiment.add_tag("metadata")

##Train
#Train see config.yml for tfrecords path with weighted classes in cross entropy
with experiment.context_manager("metadata"):
    model.read_data(mode="metadata")
    class_weight = model.calc_class_weight()    
    model.train(submodel="metadata", experiment=experiment, class_weight=class_weight)