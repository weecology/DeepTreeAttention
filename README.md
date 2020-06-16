# DeepTreeAttention
[![Build Status](https://travis-ci.org/weecology/DeepTreeAttention.svg?branch=master)](https://travis-ci.org/weecology/DeepTreeAttention)

Implementation of Hang et al. 2020 "Hyperspectral Image Classification with Attention Aided CNNs" for tree species prediction 

# Model Architecture

![](www/model.png)

## Config file

See conf/config.yml for training parameters.

# Status and Roadmap

- [x] Model skeleton with CNNs
- [x] tf.data data generators for semantic segmentation of input raster
- [x] Initial tests of backbone (see experiments)
- [ ] Add attention layers
- [ ] Recreate Houston 2018 Results
- [ ] tf.data generators for variable sized (?) pre-cropped images
- [ ] Multi-gpu

# Experiments

This repo is being tested as an open source project on comet_ml. Comet is a great machine learning dashboard. The project link is [here](https://www.comet.ml/bw4sz/deeptreeattention/view/new)
Major milestones will be listed below, but for fine-grained information on code, model structure and evaluation statistics, see individual comet pages.

# Citation

The original paper is 

* Hang, Renlong, Zhu Li, Qingshan Liu, Pedram Ghamisi, and Shuvra S. Bhattacharyya. 2020. “Hyperspectral Image Classification with Attention Aided CNNs,” May. http://arxiv.org/abs/2005.11977.
 
