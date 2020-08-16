
#Testing script that lives outside of pytest due to DeepForest dependency in a new conda env. PYTHONPATH manually added to root dir for relative paths.
import os
import pytest
import tensorflow
import numpy as np
from matplotlib import pyplot as plt
import geopandas as gpd

import prepare_field_data
from DeepTreeAttention.generators import boxes
from deepforest import deepforest

data_dir = os.path.dirname(prepare_field_data.__file__)
data_path = "{}/test_data/sample.shp".format(data_dir)
rgb_pool = "{}/test_data/rgb/*.tif".format(data_dir)
hyperspec_pool = "{}/test_data/HSI/*.tif".format(data_dir)
hyperspectral_savedir = "{}/test_data/HSI/".format(data_dir)

height = 100
width = 100

def test_process_plot():
    df = gpd.read_file(data_path)
    
    deepforest_model = deepforest.deepforest()
    deepforest_model.use_release()
    
    merged_boxes = prepare_field_data.process_plot(plot_data=df, rgb_pool=rgb_pool, deepforest_model=deepforest_model)
    assert df.shape[0] == merged_boxes.shape[0]
    
def test_run():
    df = gpd.read_file(data_path)
    
    plot_crops, plot_labels, plot_box_index = prepare_field_data.run(plot=df.plotID[0], df=df, rgb_pool=rgb_pool, hyperspectral_pool=hyperspec_pool, 
                                                          sensor="rgb", extend_box=1, hyperspectral_savedir=hyperspectral_savedir) 
    
    assert len(plot_labels) == df.shape[0]
    assert len(plot_crops) == df.shape[0]
    assert len(plot_box_index) == df.shape[0]
    
    #all indices should be unique
    assert len(np.unique(plot_box_index)) == 3
    
    
def test_main():
    created_records = prepare_field_data.main(
        field_data=data_path,
        hyperspectral_pool=hyperspec_pool,
        height=height,
        width=width,
        rgb_pool=rgb_pool,
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
            try:
                image, label = sess.run(next_element)
                assert image.shape == (1, height, width, 3)
                assert label.shape  == (1,2)
                
                plt.imshow(image[0].astype("uint8"))                
                labels.append(label)
                counter+=1
            except Exception as e:
                print(e)
                break
    assert counter ==3 
    assert max([np.argmax(x) for x in labels])

#Run tests
test_process_plot()
test_run()
test_main()