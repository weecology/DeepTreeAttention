#Experiment
import glob
import os
from comet_ml import Experiment
from DeepTreeAttention.main import AttentionModel
from DeepTreeAttention.utils import metrics
from DeepTreeAttention.visualization import visualize
from datetime import datetime
import matplotlib.pyplot as plt

experiment = Experiment(project_name="deeptreeattention", workspace="bw4sz")

#timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment.log_parameter("timestamp",timestamp)

#Create a class and run
model = AttentionModel()
model.create(name="single_conv")
model.read_data()
    
#Log config
experiment.log_parameters(model.config["train"])
experiment.log_parameter("Training Batch Size", model.config["train"]["batch_size"])

##Train
#Train see config.yml for tfrecords path
model.train()

print("Training Complete")

##Evaluate
#Evaluation scores, see config.yml for tfrecords path
train_records = glob.glob(model.config["train"]["tfrecords"] + "*.tfrecords")
y_pred, y_true = model.evaluate(model.train_records, batch_size=200)

print("get f1scores")
#F1 scores
y_true_integer = np.argmax(y_true)
y_pred_integer = np.argmax(y_pred)
micro, macro, weighted= metrics.f1_scores(y_true_integer, y_pred_integer)
experiment.log_metric("MicroF1",micro)
experiment.log_metric("MacroF1",macro)
experiment.log_metric("WeightedF1", weighted)

#Confusion matrix
class_labels = {
    0 : "Unclassified",
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

confusion_matrix = metrics.confusion(y_true, y_pred, num_classes=model.config["train"]["classes"])
experiment.log_confusion_matrix(y_true = y_true, y_predicted = y_pred, labels=list(class_labels.values()), title="Confusion Matrix")

##Predict##
predict_tfrecords = glob.glob("/orange/ewhite/b.weinstein/Houston2018/tfrecords/predict/*.tfrecord")

#Predicted raster
results = model.predict(predict_tfrecords, batch_size=200)
predicted_raster = visualize.create_raster(results)

save_dir = "{}/{}".format("/orange/ewhite/b.weinstein/Houston2018/snapshots/",timestamp)
os.mkdir(save_dir)
prediction_path = "{}/predicted_raster.png".format(save_dir)
experiment.log_image(name="Prediction", image_data=predicted_raster, image_colormap=visualize.discrete_cmap(20, base_cmap="jet"))

#Save model
model.model.save("{}/{}.h5".format(save_dir,timestamp))
