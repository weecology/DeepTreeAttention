
#Testing script that lives outside of pytest due to DeepForest dependency in a new conda env. PYTHONPATH manually added to root dir for relative paths.
import os
import pytest
import tensorflow
import numpy as np
from matplotlib import pyplot as plt
import geopandas as gpd
import glob

import prepare_field_data
from DeepTreeAttention.generators import boxes
from deepforest import deepforest

data_dir = os.path.dirname(prepare_field_data.__file__)
data_path = "{}/test_data/sample.shp".format(data_dir)
rgb_dir = "{}/test_data/rgb/*.tif".format(data_dir)
hyperspectral_dir = "{}/test_data/HSI/*.tif".format(data_dir)
hyperspectral_savedir = "{}/test_data/HSI/".format(data_dir)

hyperspectral_pool = glob.glob(hyperspectral_dir, recursive=True)
rgb_pool = glob.glob(rgb_dir, recursive=True)

height = 100
width = 100

def test_process_plot():
    df = gpd.read_file(data_path)
    
    deepforest_model = deepforest.deepforest()
    deepforest_model.use_release()
    
    merged_boxes = prepare_field_data.process_plot(plot_data=df, rgb_pool=rgb_pool, deepforest_model=deepforest_model)
    assert df.shape[0] <= merged_boxes.shape[0]
    
def test_run():
    df = gpd.read_file(data_path)
    
    plot_crops, plot_labels, plot_sites, plot_elevations, plot_box_index = prepare_field_data.run(
        plot=df.plotID[0],
        df=df,
        rgb_pool=rgb_pool,
        hyperspectral_pool=hyperspectral_pool,
        sensor="rgb",
        extend_box=1,
        hyperspectral_savedir=hyperspectral_savedir
    ) 
    
    #all have same length
    lists = [plot_crops, plot_labels, plot_sites, plot_elevations, plot_box_index]
    assert len({len(i) for i in lists}) == 1
    
    #all indices should be unique
    #assert len(np.unique(plot_box_index)) == len(plot_box_index)
    
def test_main():
    created_records = prepare_field_data.main(
        field_data=data_path,
        hyperspectral_dir=hyperspectral_dir,
        height=height,
        width=width,
        rgb_dir=rgb_dir,
        sensor="rgb",
        hyperspectral_savedir=hyperspectral_savedir,
        use_dask=False,
        extend_box=3)
    
    dataset = boxes.tf_dataset(created_records, batch_size=1)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    
    with tensorflow.Session() as sess:
        labels = []
        counter=0        
        while True:
            data, label = sess.run(next_element)
            assert data[0].shape == (1, height, width, 3)
            assert label.shape  == (1,1)
            
            plt.imshow(data[0].astype("uint8"))                
            labels.append(label)
            counter+=1
            
    assert counter==5 

#Run tests
test_process_plot()
test_run()
test_main()