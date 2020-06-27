# DeepTreeAttention
[![Build Status](https://travis-ci.org/weecology/DeepTreeAttention.svg?branch=master)](https://travis-ci.org/weecology/DeepTreeAttention)

Implementation of Hang et al. 2020 "Hyperspectral Image Classification with Attention Aided CNNs" for tree species prediction 

# Model Architecture

![](www/model.png)

## Config file

See conf/config.yml for training parameters.

# Organization

```
├── conf                   # Config files for model training and evaluation
├── data                   #  Location to place data for model reading. Most data is too large to be in version control, see below
├── DeepTreeAttention                   # Source files
├── references                    # Documentation files 
├── models                    # Trained snapshots
├── docs                   #
├── tests                    # Automated pytest tests
├── www                   # repo images
├── LICENSE
└── README.md
└── environment.yml # Conda Environment for model training and tests
```

# Roadmap

- [x] Model skeleton with CNNs
- [x] tf.data data generators for semantic segmentation of input raster
- [x] Initial tests of backbone (see experiments)
- [ ] Add attention layers
- [ ] Recreate Houston 2018 Results
- [ ] tf.data generators for variable sized (?) pre-cropped images
- [ ] Multi-gpu

# Experiments

This repo is being tested as an open source project on comet_ml. Comet is a great machine learning dashboard. The project link is [here](https://www.comet.ml/bw4sz/deeptreeattention/view/new)
Major milestones will be listed below, but for fine-grained information on code, model structure and evaluation statistics, see individual comet pages. To recreate experiments, make sure to set your own comet_ml api key by creating a .comet.config in your home directory (see https://www.comet.ml/docs/python-sdk/advanced/
).

## Data

* The Houston 2018 IEEE competition data can be downloaded [here](https://hyperspectral.ee.uh.edu/?page_id=1075) and should be placed in data/raw. This folder is not under version control. To process the sensor data into same extent as the ground truth labels run from top dir:

```
python crop_Houston2018.py
```

Which will save the crop in data/processed.


# Citation

* Hang, Renlong, Zhu Li, Qingshan Liu, Pedram Ghamisi, and Shuvra S. Bhattacharyya. 2020. “Hyperspectral Image Classification with Attention Aided CNNs,” May. http://arxiv.org/abs/2005.11977.
 
This repo can be cited on Zenodo once a release is created. 
