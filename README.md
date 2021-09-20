DeepTreeAttention
==============================

[![Github Actions](https://github.com/Weecology/DeepTreeAttention/actions/workflows/pytest.yml/badge.svg)](https://github.com/Weecology/DeepTreeAttention/actions/)

Tree Species Prediction for the National Ecological Observatory Network (NEON)

Implementation of Hang et al. 2020 [Hyperspectral Image Classification with Attention Aided CNNs](https://arxiv.org/abs/2005.11977) for tree species prediction.

# Model Architecture

![](www/model.png)

# Performance

| Model | Crown Micro | Crown Macro | Experiment    |
| ----- | ----------- | ----------- | --- |
| Vanilla CNN  | 33.2        | 26.2        |   [click here](https://www.comet.ml/bw4sz/deeptreeattention/5a1ad2d55095413d9fcfb67875fed1b2?experiment-tab=metrics)  |

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── environment.yml   <- Conda requirements
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py       <- Makes src a Python module
    │   ├── data.py           <- Pytorch Lighting data module for creating dataloaders for model training
    │   ├── generate.py       <- Convert csv of point files to tree crowns
    │   ├── main.py           <- Pytorch Lightning Module for model training
    │   ├── neon_paths.py     <- Utilities for getting paths and metadata from NEON HSI data
    │   ├── patches.py        <- Convert tree crowns into a set of pixels with overlapping windows
    │   ├── start_cluster.py  <- dask utilities for SLURM parallel processing
    │   ├── CHM.py            <- Canopy Height Model Lidar Processing
    │   ├── Hyperspectral.py  <- Hyperspectral conversion from .h5 to .tif
    │   ├── Models         <- Model Architectures

--------

# Workflow
There are three main parts to this project, a 1) data module, a 2) model module, and 3) a trainer module. Usually the data_module is created to hold the train and test split and keep track of data generation reproducibility. Then a model architecture is created and pass to the model module along with the data module. Finally the model module is passed to the trainer.

```
#1) 
data_module = data.TreeData(csv_file="data/raw/neon_vst_data_2021.csv", regenerate=False, client=client)

#2)
model = <create a pytorch NN.module>
m = main.TreeModel(model=model, bands=data_module.config["bands"], classes=data_module.num_classes,label_dict=data_module.species_label_dict)

#3
trainer = Trainer()
trainer.fit(m, datamodule=data_module)
```

## Pytorch Lightning Data Module (data.TreeData)

This repo contains a pytorch lightning data module for reproducibility. The goal of the project is to make it easy to share with others within our research group, but we welcome contributions from outside the community. While all data is public, it is VERY large (>20TB) and cannot be easily shared. If you want to reproduce this work, you will need to download the majority of NEON's camera, HSI and CHM data and change the paths in the config file. For the 'raw' NEON tree stem data see data/raw/neon_vst_2021.csv. The data module starts from this state, which are x,y locations for each tree. It then performs the following actions as an end-to-end workflow.

1. Filters the data to represent trees over 3m with sufficient number of training samples
2. Extract the LiDAR derived canopy height and compares it to the field measured height. Trees that are below the canopy are excluded based on the min_CHM_diff parameter in the config.
3. Splits the training and test x,y data such that field plots are either in training or test.
4. For each x,y stem location the crown is predicted by the tree detection algorithm (DeepForest - https://deepforest.readthedocs.io/).
5. Crops of each tree crown are created and divided into pixel windows for pixel-level prediction.

This workflow does not need to be run on every experiment. If you are satisifed with the current train/test split and data generation process, set regenerate=False

```
data_module = data.TreeData(csv_file="data/raw/neon_vst_data_2021.csv", regenerate=False)
data_module.setup()
```

## Pytorch Lightning Training Module (data.TreeModel)

Training is handled by the TreeModel class which loads a model from the models folder, reads the config file and runs the training. The evaluation metrics and images are computed and put of the comet dashboard

```
m = main.TreeModel(model=Hang2020.vanilla_CNN, bands=data_module.config["bands"], classes=data_module.num_classes,label_dict=data_module.species_label_dict)

trainer = Trainer(
    gpus=data_module.config["gpus"],
    fast_dev_run=data_module.config["fast_dev_run"],
    max_epochs=data_module.config["epochs"],
    accelerator=data_module.config["accelerator"],
    logger=comet_logger)
   
trainer.fit(m, datamodule=data_module)
```

## Evaluation metrics

The training metrics are computed at the pixel level during training. At the end of training, predictions are made for each crown using majority rule among pixels. The crown-level accuracy is then computed across classes.

Current metrics include micro/macro pixel accuracy and micro/macro crown accuracy. Pixel accuracy is the proportion of pixel crops correctly predicted, crown accuracy is aggregate predicted label, for example using majority rule for the entire crown's worth of pixels.

### Dev Guide

In general, major changes or improvements should be made on a new git branch. Only core improvements should be made on the main branch. If a change leads to higher scores, please create a pull request. Any pull requests are expected to have pytest unit tests (see tests/) that cover major use cases.

## Model Architectures

The TreeModel class takes in a create model function

```
m = main.TreeModel(model=Hang2020.vanilla_CNN)
```

Any model can be specified provided it follows the following input and output arguments

```
class myModel(Module):
    """
    Model description
    """
    def __init__(self, bands, classes):
        super(myModel, self).__init__()
        <define model architecture here>

    def forward(self, x):
        <forward method for computing loss goes here>
        class_scores = F.softmax(x)
        
        return class_scores
```

to create a model that takes in new inputs, I strongly recommend sub-classing the existing TreeData and TreeModel classes and extending them. For an example, see the MetadataModel in models/metadata.py

