#Experiment
from comet_ml import Experiment
from datetime import datetime
from DeepTreeAttention.trees import AttentionModel
from DeepTreeAttention.utils import metrics, resample, start_cluster
from DeepTreeAttention.visualization import visualize
from tensorflow.keras import metrics as keras_metrics
from random import randint
from time import sleep
from distributed import wait

import glob
import os
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
    experiment = Experiment(project_name="neontrees", workspace="bw4sz")

    #Create output folder
    #Sleep for a moment to allow queries to build up in SLURM queue
    sleep(randint(0,10))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = "{}/{}".format("/orange/idtrees-collab/DeepTreeAttention/snapshots/",timestamp)
    os.mkdir(save_dir)
    
    experiment.log_parameter("timestamp",timestamp)
    
    #Create a class and run
    model = AttentionModel(config="/home/b.weinstein/DeepTreeAttention/conf/tree_config.yml")
    model.create()
        
    #Log config
    experiment.log_parameters(model.config["train"])
    experiment.log_parameters(model.config["evaluation"])    
    experiment.log_parameters(model.config["predict"])
    
    ##Train
    #Train see config.yml for tfrecords path with weighted classes in cross entropy
    model.read_data()
    class_weight = model.calc_class_weight(model.val_split)
    
    ## Train subnetwork
    experiment.log_parameter("Train subnetworks", True)
    with experiment.context_manager("spatial_subnetwork"):
        print("Train spatial subnetwork")
        model.read_data(mode="submodel")
        model.train(submodel="spatial", class_weight=[class_weight, class_weight, class_weight])
    with experiment.context_manager("spectral_subnetwork"):
        print("Train spectral subnetwork")    
        model.read_data(mode="submodel")   
        model.train(submodel="spectral", class_weight=[class_weight, class_weight, class_weight])
            
    #Train full model
    experiment.log_parameter("Class Weighted", True)
    model.read_data()
    model.train(class_weight=class_weight, experiment=experiment)
    
    #Get Alpha score for the weighted spectral/spatial average. Higher alpha favors spatial network.
    if model.config["train"]["weighted_sum"]:
        estimate_a = model.model.layers[-1].get_weights()
        experiment.log_metric(name="spatial-spectral weight", value=estimate_a[0][0])
        
    ##Evaluate
    #Evaluation scores, see config.yml for tfrecords path
    y_pred, y_true = model.evaluate(model.val_split)
    
    #Evaluation accuracy
    #Something is wrong here, this does not seem right,.
    eval_acc = keras_metrics.CategoricalAccuracy()
    eval_acc.update_state(y_true, y_pred)
    experiment.log_metric("Evaluation Accuracy",eval_acc.result().numpy())
    
    macro, micro = metrics.f1_scores(y_true, y_pred)
    experiment.log_metric("MicroF1",micro)
    experiment.log_metric("MacroF1",macro)
    
    print("Unique labels in ytrue {}, unique labels in y_pred {}".format(np.unique(np.argmax(y_true,1)),np.unique(np.argmax(y_pred,1))))
    
    #Read class labels
    labeldf = pd.read_csv(model.classes_file)    
    experiment.log_confusion_matrix(y_true = y_true, y_predicted = y_pred, labels=list(labeldf.taxonID.values), title="Confusion Matrix")
    
    #Save model
    model.model.save("{}/{}.h5".format(save_dir,timestamp))


    
    
