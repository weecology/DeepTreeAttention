#Test multinomial 
import pytest
import geopandas as gpd
import pandas as pd
from shapely import geometry
from src import multonomial

def test_run(tmpdir):
    #Create predictions
    boxes = [geometry.box(0, 0, 5, 5),geometry.box(0, 0, 5, 5),geometry.box(0, 0, 5, 5), geometry.box(0, 0, 5, 5)]
    p = gpd.GeoDataFrame({"ensemble_label":[0,0,1,None],"ensembleTa":["A","A","B","DEAD"],"ensemble_score":[0.9999,0.05,0.9999,None],"geometry":boxes})
    p.to_file("{}/example.shp".format(tmpdir))
    
    #Create confusion matrix
    confusion = pd.DataFrame({"A":[0.85,0.15],"B":[0.15,0.85]})
    confusion["predicted"] = ["A","B"]
    confusion.to_csv("{}/confusion.csv".format(tmpdir), index=False)
    
    counts = multonomial.run(tile="{}/example.shp".format(tmpdir), confusion_path="{}/confusion.csv".format(tmpdir))
