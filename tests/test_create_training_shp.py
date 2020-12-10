from DeepTreeAttention.generators import create_training_shp
from DeepTreeAttention.trees import __file__ as ROOT
import os
import pytest
import geopandas as gpd
from shapely.geometry import Point

@pytest.fixture()
def testdata():
    path = "data/raw/test_with_uid.csv"
    field_data_path = "data/raw/2020_vst_december.csv"    
    shp = create_training_shp.test_split(path, field_data_path)
    
    assert not shp.empty
    
    return shp
    
def test_train_test_split():
    path = "data/raw/2020_vst_december.csv"
    shp = create_training_shp.train_test_split(os.path.dirname(os.path.dirname(ROOT)), debug=True)
    assert not shp.empty    
    
#def test_filter_CHM():
    #lookup_glob = "data/raw/*CHM*"
    
    ##Create fake data, within one good point and one bad point by looking at the CHM -> true value is 12.43
    #good_point = {"geometry":Point(320224.5, 4881567.7), "height": 12, "plotID":"BART_000"}
    #bad_point = {"geometry":Point(320224.5, 4881567.7), "height": 0, "plotID": "BART_000"}
    #null_point = {"geometry":Point(320224.5, 4881567.7), "height": None, "plotID": "BART_000"}
    
    #traindata = gpd.GeoDataFrame([good_point, bad_point, null_point])
    #filtered_data = create_training_shp.filter_CHM(traindata, lookup_glob=lookup_glob, min_diff=2)
    #assert filtered_data.shape[0] == 2
    #assert list(filtered_data.height.astype(int).values) == [12,12]