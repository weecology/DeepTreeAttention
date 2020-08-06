#Predict species class for a set of DeepForest boxes
from distributed import wait
import argparse
import geopandas
from DeepTreeAttention.trees import AttentionModel
from DeepTreeAttention.generators import boxes
from DeepTreeAttention.utils import start_cluster
import glob
import os
