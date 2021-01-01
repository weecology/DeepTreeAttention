from DeepTreeAttention.generators import create_training_shp
from DeepTreeAttention.trees import __file__ as ROOT
import os
import pytest

ROOT = os.path.dirname(os.path.dirname(ROOT))

@pytest.fixture()
def testdata():
    path = "{}/data/raw/test_with_uid.csv".format(ROOT)
    field_data_path = "{}/data/raw/2020_vst_december.csv".format(ROOT)    
    shp = create_training_shp.test_split(path, field_data_path)
    
    assert not shp.empty
    
    return shp
    
def test_train_test_split():
    create_training_shp.train_test_split(ROOT, debug=True)