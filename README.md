DeepTreeAttention
==============================

Tree Species Prediction for the National Ecological Observatory Network (NEON)

## Road map

- [] Data Generation: Convert NEON field data into crowns -> pixels for mapping
- [] Baseline Model: A vanilla 2D CNN score for computing comet environment, metrics and figures
- [] 3D CNN with Attention

After this there are many different routes including weak label learning, self-supervised contrastive learning and other psuedo-labeling approaches

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── environment.yml   <- Conda requirements
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   ├── data           <- Pytorch Lighting data module for creating dataloaders for model training
    │   ├── dataset        <- Pytorch dataset for generating batches of data
    │   ├── generate       <- Convert csv of point files to tree crowns
    │   ├── main           <- Pytorch Lightning Module for model training
    │   ├── neon_paths     <- Utilities for getting paths and metadata from NEON HSI data
    │   ├── patches        <- Convert tree crowns into a set of pixels with overlapping windows
    │   ├── start_cluster  <- dask utilities for SLURM parallel processingt


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
