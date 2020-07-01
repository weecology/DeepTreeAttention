#Experiment
import glob
import os
from comet_ml import Experiment
from DeepTreeAttention.main import AttentionModel
from DeepTreeAttention.visualization.visualize import discrete_cmap
from datetime import datetime

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

model.train()

predict_tfrecords = glob.glob("/orange/ewhite/b.weinstein/Houston2018/tfrecords/predict/*.tfrecord")
predicted_raster = model.predict(predict_tfrecords, batch_size=128)
experiment.log_image(predicted_raster, "Direct logged prediction")

save_dir = "{}/{}".format("/orange/ewhite/b.weinstein/Houston2018/snapshots/",timestamp)
os.makedir(save_dir)
prediction_path = "{}/predicted_raster.png".format(save_dir)

#Save color map
plt.imsave(prediction_path, predicted_raster, cmap=discrete_cmap(20))
experiment.log_image("Colored Prediction", prediction_path)

#Save model
model.model.save("{}/{}.h5".format(save_dir,timestamp))
