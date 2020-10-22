#test species id boxes
import pytest
import os
from DeepTreeAttention.generators import boxes
import geopandas as gpd
import numpy as np

#random label predictions just for testing
test_predictions = "data/raw/2019_BART_5_320000_4881000_image.shp"

#Use a small rgb crop as a example tile
test_sensor_tile = "data/raw/2019_BART_5_320000_4881000_image_crop.tif"

@pytest.mark.parametrize("train",[True, False])
def test_generate_tfrecords(train, tmpdir):
    
    shp = gpd.read_file(test_predictions)
    created_records = boxes.generate_tfrecords(
        shapefile=test_predictions,
        site = 1,
        elevation=100,
        heights=np.random.random(shp.shape[0])*10,
        savedir=tmpdir,
        train=train,
        HSI_sensor_path=test_sensor_tile,
        RGB_sensor_path=test_sensor_tile,
        species_label_dict=None,
        RGB_size=100,
        HSI_size=20,
        classes=6,
        number_of_sites=10)
    
    assert all([os.path.exists(x) for x in created_records])
    
    if train:
        dataset = boxes.tf_dataset(created_records, batch_size=2, mode="ensemble")
    else:
        dataset = boxes.tf_dataset(created_records, batch_size=2, mode="predict")
    
    if train:
        #Yield a batch of data and confirm its shape
        for (HSI, RGB, elevation, height, site), label_batch in dataset.take(3):
            assert HSI.shape == (2,20,20,3)
            assert RGB.shape == (2,100,100,3)    
            assert elevation.shape == (2)
            assert site.shape == (2,10)            
            assert height.shape == (2)                       
            assert label_batch.shape == (2,6)
    else:
        for (HSI, RGB ), box_index_batch in dataset.take(3):
            assert HSI.shape == (2,20,20,3)
            assert RGB.shape == (2,100,100,3) 
            assert box_index_batch.shape == (2,)

def test_metadata(tmpdir):
    shp = gpd.read_file(test_predictions)
    
    created_records = boxes.generate_tfrecords(
        shapefile=test_predictions,
        site = 1,
        heights=np.random.random(shp.shape[0])*10,        
        elevation=100,
        savedir=tmpdir,
        HSI_sensor_path=test_sensor_tile,
        RGB_sensor_path=test_sensor_tile,
        species_label_dict=None,
        RGB_size=100,
        HSI_size=20,
        classes=6,
        number_of_sites=10)
    
    dataset = boxes.tf_dataset(created_records, batch_size=2, mode="metadata")
    for data, label_batch in dataset.take(3):
        elevation, height, site = data
        assert elevation.numpy().shape == (2,)
        assert site.numpy().shape == (2,10)
