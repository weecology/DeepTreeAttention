# DeepTreeAttention
[![Build Status](https://travis-ci.org/weecology/DeepTreeAttention.svg?branch=master)](https://travis-ci.org/weecology/DeepTreeAttention)

Implementation of Hang et al. 2020 [Hyperspectral Image Classification with Attention Aided CNNs](https://arxiv.org/abs/2005.11977) for tree species prediction. This repo first reproduces the work using the Houston2018 open source benchmark and then applies it to NEON trees. The two workflows differ in that the goal of the Houston2018 is to predict every cell in a raster, whereas the NEON trees are first predicted based on object detection. The goal is then to predict a single label per predicted tree box.

# Model Architecture

![](www/model.png)


# Organization

```
├── conf                   # Config files for model training and evaluation
├── data                   #  Location to place data for model reading. Most data is too large to be in version control, see below
├── DeepTreeAttention                   # Source files
├── experiments                    # Model training and SLURM multi-gpu cluster experiments with comet dashboards 
├── models                    # Trained snapshots
├── docs                   #
├── tests                    # Automated pytest tests
├── www                   # repo images
├── LICENSE
└── README.md
└── environment.yml # Conda Environment for model training and tests
```

# Roadmap

## Houston2018

- [x] Model skeleton with CNNs
- [x] tf.data data generators for semantic segmentation of input raster
- [x] Initial tests of backbone (see experiments)
- [x] Add attention layers
- [x] Multi-gpu

## NEON

- [x] Data pipeline to predict species class for a DeepForest bounding box (https://deepforest.readthedocs.io/) for NEON Woody Veg Data
- [x] Data pipeline to predict species class for a bounding box with weakly learned labels from random forest
- [x] Training Pipeline for Hyperspectral DeepTreeAttention Model
- [ ] Training Pipeline for RGB DeepTreeAttention Model
- [ ] Training Pipeline for LiDAR classification model (PointNet Variant)
- [ ] Learned fusion among data inputs

# How to view the experiments

This repo is being tested as an open source project on comet_ml. Comet is a great machine learning dashboard. The project link is [here](https://www.comet.ml/bw4sz/deeptreeattention/view/new).
Major milestones will be listed below, but for fine-grained information on code, model structure and evaluation statistics, see individual comet pages. To recreate experiments, make sure to set your own comet_ml api key by creating a .comet.config in your home directory (see https://www.comet.ml/docs/python-sdk/advanced/
).

## Houston2018

* [Backbone CNN](https://www.comet.ml/bw4sz/deeptreeattention/d32b066dce254c2d9742331e97b494f5?experiment-tab=chart&showOutliers=true&smoothing=0&view=Hang&xAxis=step)
* [With Attention Layers](https://www.comet.ml/bw4sz/deeptreeattention/15d3246de6cf490cacf63d1764c9c494?experiment-tab=chart&showOutliers=true&smoothing=0&view=Hang&xAxis=step)
* [With Ensemble Training of Spectral/Spatial subnetworks on multi-gpu](https://www.comet.ml/bw4sz/deeptreeattention/5d632e11d2484bfdb4de32166c5099ce)

### Config file

See conf/houston_config.yml for training parameters.

### Data

* The Houston 2018 IEEE competition data can be downloaded [here](https://hyperspectral.ee.uh.edu/?page_id=1075) and should be placed in data/raw. This folder is not under version control. 

### Workflow

To process the sensor data into same extent as the ground truth labels run from top dir:

```
python experiments/Houston2018/crop_Houston2018.py
```

Which will save the crop in data/processed.

This repo uses tfrecords to feed into tf.keras to optomize GPU performance. Records need to be created before model training.

```
python experiments/Houston2018/generate.py
```

and then run training and log results to the comet dashboard.

```
python experiments/Houston2018/run_Houston2018.py
```

## NEON

### Config file

See conf/tree_config.yml for training parameters.

### Data

The field data are from NEON's woody vegetation structure dataset. I curated .shp is found at data/processed/field.shp which contains species labels and utm coordinates of each tree stem

### Workflow

To generate training data from existing shapefiles of deepforest predictions

```
python experiments/Trees/generate.py
```

To generate new deepforest boxes, you will need to create a seperate conda environment. DeepForest requires tensorflow <2.0 where this repo is >2.0. The requirements are otherwise the same. To generate boxes see

```
python experiments/Trees/prepare_field_data.py
```

After creating training data the main entry point is 

```
python experiments/Trees/run.py
```

# Citation

* Hang, Renlong, Zhu Li, Qingshan Liu, Pedram Ghamisi, and Shuvra S. Bhattacharyya. 2020. “Hyperspectral Image Classification with Attention Aided CNNs,” May. http://arxiv.org/abs/2005.11977.
 
This repo can be cited on Zenodo once a release is created. 
