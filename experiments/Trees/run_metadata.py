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
model.read_data(mode="metadata")

#Cree 
inputs, outputs = metadata.metadata_model(classes=74)
meta_model = tf.keras.Model(inputs=inputs,
                                        outputs=outputs,
                                        name="DeepTreeAttention")

meta_model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=["acc"]
)

labeldf = pd.read_csv(model.classes_file)
callback_list = callbacks.create(
    log_dir=model.log_dir,
    experiment=experiment,
    validation_data=model.val_split,
    label_names=list(labeldf.taxonID.values),
    submodel=None)

meta_model.fit(model.train_split,
    epochs=model.config["train"]["epochs"],
    validation_data=model.val_split,
    callbacks=callback_list
)
