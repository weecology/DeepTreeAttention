from DeepTreeAttention.generators import create_training_shp
import os
import pytest

@pytest.fixture()
def testdata():
    path = "data/raw/test_with_uid.csv"
    field_data_path = "data/raw/2020_vst_december.csv"  
    shp = create_training_shp.test_split(path, field_data_path)
    
    assert not shp.empty
    
    return shp
    
def test_train_test_split():
    create_training_shp.train_test_split(".", debug=True)