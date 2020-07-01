#Experiment
import glob
from matplotlib.pyplot import imshow
from comet_ml import Experiment
from DeepTreeAttention.main import AttentionModel

experiment = Experiment(project_name="deeptreeattention", workspace="bw4sz")

#Create a class and run
model = AttentionModel()
model.create(name="single_conv")
model.read_data()
    
#Log config
experiment.log_parameters(model.config["train"])
experiment.log_parameter("Training Batch Size", model.config["train"]["batch_size"])

model.train()

predict_tfrecords = glob.glob("/orange/ewhite/b.weinstein/Houston2018/tfrecords/predict/*.tfrecord")
predicted_raster = model.predict(predict_tfrecords, batch_size=128)
fig = imshow(predicted_raster)
experiment.log_figure(figure=fig,figure_name="Predicted Raster")

