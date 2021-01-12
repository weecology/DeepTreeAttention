#test species id boxes
import pytest
import os
import geopandas as gpd
import numpy as np
import tensorflow as tf

from DeepTreeAttention.generators import boxes
from DeepTreeAttention.models import Hang2020_geographic, metadata

#random label predictions just for testing
test_predictions = "data/raw/2019_BART_5_320000_4881000_image_small.shp"

#Use a small rgb crop as a example tile
test_sensor_tile = "data/raw/2019_BART_5_320000_4881000_image_crop.tif"

test_hsi_tile = "data/raw/2019_BART_5_320000_4881000_image_hyperspectral_crop.tif"

@pytest.fixture()
def ensemble_model():
    sensor_inputs, sensor_outputs, spatial, spectral = Hang2020_geographic.define_model(classes=2, height=20, width=20, channels=369)    
    model1 = tf.keras.Model(inputs=sensor_inputs, outputs=sensor_outputs)
    
    metadata_model = metadata.create(classes=2, sites=10, domains =10, learning_rate=0.001)
    ensemble_model = Hang2020_geographic.learned_ensemble(HSI_model=model1, metadata_model=metadata_model, classes=2)
    ensemble_model = tf.keras.Model(ensemble_model.inputs, ensemble_model.get_layer("submodel_concat").output)
    
    return ensemble_model
    
@pytest.fixture()
def created_records(tmpdir, ensemble_model):
    shp = gpd.read_file(test_predictions)    
    created_records = boxes.generate_tfrecords(
        shapefile=test_predictions,
        domain=1,
        site = 1,
        elevation=100.0,
        savedir=tmpdir,
        HSI_sensor_path=test_hsi_tile,
        RGB_sensor_path=test_sensor_tile,
        species_label_dict=None,
        RGB_size=100,
        HSI_size=20,
        classes=6,
        number_of_sites=10,
        number_of_domains=10,
        ensemble_model=ensemble_model,
        raw_boxes=test_predictions
    )
    
    return created_records

def test_generate_records(tmpdir, ensemble_model):
    created_records = boxes.generate_tfrecords(
        shapefile=test_predictions,
        domain=1,
        site = 1,
        elevation=100.0,
        savedir=tmpdir,
        HSI_sensor_path=test_hsi_tile,
        RGB_sensor_path=test_sensor_tile,
        species_label_dict=None,
        RGB_size=100,
        HSI_size=20,
        classes=6,
        number_of_sites=10,
        number_of_domains=10,
        ensemble_model=ensemble_model,
        raw_boxes=test_predictions
    )
    
    assert len(created_records) > 0 
    
@pytest.mark.parametrize("train",[True, False])
def test_tf_dataset(train, created_records):
    assert all([os.path.exists(x) for x in created_records])
    
    if train:
        dataset = boxes.tf_dataset(created_records, mode = "RGB", batch_size=2)
    else:
        dataset = boxes.tf_dataset(created_records, mode="RGB", batch_size=2, ids = True)
    
    if train:
        #Yield a batch of data and confirm its shape
        for batch in dataset.take(1):
            data, label = batch
            assert data.shape == (2,100,100,3)    
            assert label.shape == (2,6)
    else:
        for ids, batch in dataset.take(3):
            data, label = batch
            assert data.shape == (2,100,100,3)    
            assert ids.shape == (2)
            
def test_metadata(created_records):    
    dataset = boxes.tf_dataset(created_records, batch_size=2, mode="metadata")
    for data, label_batch in dataset.take(1):
        elevation, site, domain = data
        assert elevation.numpy().shape == (2,)
        assert site.numpy().shape == (2,10)
        assert domain.numpy().shape == (2,10)
        
def test_RGB_submodel(created_records):    
    dataset = boxes.tf_dataset(created_records, batch_size=2, mode = "RGB_submodel")
    for batch in dataset.take(1):
        data, label = batch
        assert data.shape == (2,100,100,3)    
        assert len(label) == 3
        assert label[0].shape == (2,6)        

def test_ensemble(created_records):    
    dataset = boxes.tf_dataset(created_records, batch_size=2, mode="ensemble")
    for data, label_batch in dataset.take(1):
        HSI, elevation, site, domain = data
        
        assert HSI.shape == (2,20,20,369)    
        assert elevation.numpy().shape == (2,)
        assert site.numpy().shape == (2,10)
        assert domain.numpy().shape == (2,10)

def test_neighbor(created_records):    
    dataset = boxes.tf_dataset(created_records, batch_size=2, mode="neighbors")
    for data, label_batch in dataset.take(1):
        HSI, elevation, site, domain, neighbor_array = data
        
        assert HSI.shape == (2,20,20,369)    
        assert neighbor_array.shape == (2,5,4)            
        assert elevation.numpy().shape == (2,)
        assert site.numpy().shape == (2,10)
        assert domain.numpy().shape == (2,10)
        
def test_id_train(created_records):
    shp = gpd.read_file(test_predictions)        
    dataset = boxes.tf_dataset(created_records, batch_size=2, ids=True, mode = "RGB")
    for ids, batch in dataset.take(1):
        data, label = batch
        assert ids.numpy().shape == (2,)
    
    assert all([x in shp.index.values for x in ids.numpy()])