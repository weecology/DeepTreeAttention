#test species id boxes
import pytest
import os
from DeepTreeAttention.generators import boxes
import geopandas as gpd
import numpy as np

#random label predictions just for testing
test_predictions = "data/raw/2019_BART_5_320000_4881000_image_small.shp"

#Use a small rgb crop as a example tile
test_sensor_tile = "data/raw/2019_BART_5_320000_4881000_image_crop.tif"

@pytest.fixture()
def created_records(tmpdir):
    shp = gpd.read_file(test_predictions)    
    created_records = boxes.generate_tfrecords(
        shapefile=test_predictions,
        site = 1,
        heights=np.random.random(shp.shape[0])*10,        
        elevation=100.0,
        savedir=tmpdir,
        HSI_sensor_path=test_sensor_tile,
        RGB_sensor_path=test_sensor_tile,
        species_label_dict=None,
        RGB_size=100,
        HSI_size=20,
        classes=6,
        number_of_sites=10)
    
    return created_records
    
@pytest.mark.parametrize("train",[True, False])
def test_generate_tfrecords(train, created_records):
    assert all([os.path.exists(x) for x in created_records])
    
    if train:
        dataset = boxes.tf_dataset(created_records, HSI=False, batch_size=2)
    else:
        dataset = boxes.tf_dataset(created_records, HSI=False, batch_size=2, labels=False, ids = True)
    
    if train:
        #Yield a batch of data and confirm its shape
        for data, label_batch in dataset.take(3):
            RGB, height, elevation, site = data 
            assert RGB.shape == (2,100,100,3)    
            assert label_batch.shape == (2,6)
    else:
        for ids, data in dataset.take(3):
            RGB, height, elevation, site = data 
            assert RGB.shape == (2,100,100,3)    
            assert ids.shape == (2)
            
def test_metadata(created_records):    
    dataset = boxes.tf_dataset(created_records, batch_size=2, RGB=False, HSI=False)
    for data, label_batch in dataset.take(3):
        elevation, height, site = data
        assert elevation.numpy().shape == (2,)
        assert site.numpy().shape == (2,10)
        
def test_RGB_submodel(created_records):    
    dataset = boxes.tf_dataset(created_records, batch_size=2, RGB=True, HSI=False, metadata=False, submodel="spectral")
    for data, label_batch in dataset.take(3):
        assert data[0].shape == (2,100,100,3)    
        assert len(label_batch) == 3
        assert label_batch[0].shape == (2,6)        

def test_id_train(created_records):
    shp = gpd.read_file(test_predictions)        
    dataset = boxes.tf_dataset(created_records, batch_size=2, ids=True, HSI=False)
    for ids, data, label_batch in dataset.take(3):
        assert ids.numpy().shape == (2,)
    
    basename = os.path.splitext(os.path.basename(test_predictions))[0]
    shp["box_index"] = ["{}_{}".format(basename, x) for x in shp.index.values]
    assert all([x.decode() in shp.box_index.values for x in ids.numpy()])