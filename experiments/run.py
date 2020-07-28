#Experiment
from comet_ml import Experiment
import glob
import numpy as np
import os
from datetime import datetime
from DeepTreeAttention.main import AttentionModel
from DeepTreeAttention.utils import metrics, resample
from DeepTreeAttention.visualization import visualize
import matplotlib.pyplot as plt
from tensorflow.keras import metrics as keras_metrics

experiment = Experiment(project_name="deeptreeattention", workspace="bw4sz")

#Create output folder
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = "{}/{}".format("/orange/ewhite/b.weinstein/Houston2018/snapshots/",timestamp)
os.mkdir(save_dir)

experiment.log_parameter("timestamp",timestamp)

#Create a class and run
model = AttentionModel()
model.create()
experiment.log_parameter("Training Batch Size", model.config["train"]["batch_size"])
    
#Log config
experiment.log_parameters(model.config["train"])

##Train
#Train see config.yml for tfrecords path with weighted classes in cross entropy
model.read_data(validation_split=True)    
class_weight = model.calc_class_weight()

## Train subnetwork
experiment.log_parameter("Train subnetworks", True)
with experiment.context_manager("spatial_subnetwork"):
    print("Train spatial subnetwork")
    model.read_data(mode="submodel",validation_split=True)
    model.train(submodel="spatial", class_weight=[class_weight, class_weight, class_weight])

with experiment.context_manager("spectral_subnetwork"):
    print("Train spectral subnetwork")    
    model.read_data(mode="submodel",validation_split=True)   
    model.train(submodel="spectral", class_weight=[class_weight, class_weight, class_weight])
        
#Train full model
experiment.log_parameter("Class Weighted", True)
model.read_data(validation_split=True)
model.train(class_weight=class_weight)

#Get Alpha score for the weighted spectral/spatial average. Higher alpha favors spatial network.
if model.config["train"]["weighted_sum"]:
    estimate_a = model.model.layers[-1].get_weights()
    experiment.log_metric(name="spatial/spectral weight", value=estimate_a)
    
##Evaluate
#Evaluation scores, see config.yml for tfrecords path
y_pred, y_true = model.evaluate(model.val_split)

#Evaluation accuracy
eval_acc = keras_metrics.CategoricalAccuracy()
eval_acc.update_state(y_true, y_pred)
experiment.log_metric("Evaluation Accuracy",eval_acc.result().numpy())

macro, micro = metrics.f1_scores(y_true, y_pred)
experiment.log_metric("MicroF1",micro)
experiment.log_metric("MacroF1",macro)

#Confusion matrix
class_labels = {
    0: "Unclassified",
    1 : "Healthy grass",
    2 : "Stressed grass",
    3 : "Artificial turf",
    4 : "Evergreen trees",
    5 : "Deciduous trees",
    6 : "Bare earth",
    7 : "Water",
    8 : "Residential buildings",
    9 : "Non-residential buildings",
    10 : "Roads",
    11 : "Sidewalks",
    12 : "Crosswalks",
    13 : "Major thoroughfares",
    14 : "Highways",
    15 : "Railways",
    16 : "Paved parking lots",
    17 : "Unpaved parking lots",
    18 : "Cars",
    19 : "Trains",
    20 : "Stadium seat"
}

print("Unique labels in ytrue {}, unique labels in y_pred {}".format(np.unique(np.argmax(y_true,1)),np.unique(np.argmax(y_pred,1))))
experiment.log_confusion_matrix(y_true = y_true, y_predicted = y_pred, labels=list(class_labels.values()), title="Confusion Matrix")

#Predict
predict_tfrecords = glob.glob("/orange/ewhite/b.weinstein/Houston2018/tfrecords/predict/*.tfrecord")
results = model.predict(predict_tfrecords, batch_size=512)
#predicted classes
print(results.label.unique())

predicted_raster = visualize.create_raster(results)
print(np.unique(predicted_raster))
experiment.log_image(name="Prediction", image_data=predicted_raster, image_colormap=visualize.discrete_cmap(20, base_cmap="jet"))

#Save as tif for resampling
prediction_path = os.path.join(save_dir,"prediction.tif")
predicted_raster = np.expand_dims(predicted_raster, 0)
resample.create_tif("/home/b.weinstein/DeepTreeAttention/data/processed/20170218_UH_CASI_S4_NAD83.tif", filename=prediction_path, numpy_array=predicted_raster)
filename = resample.resample(prediction_path)
experiment.log_image(name="Resampled Prediction", image_data=filename, image_colormap=visualize.discrete_cmap(20, base_cmap="jet"))

#Save model
model.model.save("{}/{}.h5".format(save_dir,timestamp))

