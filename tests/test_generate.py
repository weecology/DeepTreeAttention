#Test generate
from src import generate
import glob
import geopandas as gpd
import pandas as pd
from deepforest import main
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

ROOT = os.path.dirname(os.path.dirname(generate.__file__))
rgb_pool = glob.glob("{}/tests/data/*.tif".format(ROOT))
rgb_path = "{}/tests/data/2019_D01_HARV_DP3_726000_4699000_image_crop.tif".format(ROOT)
data_path = "{}/tests/data/sample.shp".format(ROOT)
plot_data = gpd.read_file(data_path)        

def test_predict_trees():
    m = main.deepforest()
    m.use_release(check_release=False)
    boxes = generate.predict_trees(deepforest_model=m, rgb_path=rgb_path, bounds=plot_data.total_bounds)
    assert not boxes.empty 

def test_empty_plot():
    #DeepForest prediction
    deepforest_model = main.deepforest()
    deepforest_model.use_release(check_release=False)
    boxes = generate.predict_trees(deepforest_model=deepforest_model, rgb_path=rgb_path, bounds=plot_data.total_bounds)

    #fake offset boxes by adding a scalar to the geometry
    boxes["geometry"] = boxes["geometry"].translate(100000)
        
    #Merge results with field data, buffer on edge 
    merged_boxes = gpd.sjoin(boxes, plot_data)
    
    assert merged_boxes.empty

    #If no remaining boxes just take a box around center
    merged_boxes= generate.create_boxes(plot_data)
        
    #If there are multiple boxes, take the center box
    grouped = merged_boxes.groupby("individual")
    
    cleaned_boxes = []
    for value, group in grouped:
        choosen_box = generate.choose_box(group, plot_data)
        cleaned_boxes.append(choosen_box)
    
    merged_boxes = gpd.GeoDataFrame(pd.concat(cleaned_boxes),crs=merged_boxes.crs)
    merged_boxes = merged_boxes.drop(columns=["xmin","xmax","ymin","ymax"])
    
    assert not merged_boxes.empty
    
def test_process_plot():
    df = gpd.read_file(data_path)
    deepforest_model = main.deepforest()
    deepforest_model.use_release(check_release=False)
    
    merged_boxes, boxes = generate.process_plot(plot_data=df, rgb_pool=rgb_pool, deepforest_model=deepforest_model)
    assert df.shape[0] >= merged_boxes.shape[0]
    
def test_run(tmpdir):
    df = gpd.read_file(data_path)
    plot = df.plotID.unique()[0]
    generate.run(
        plot=plot,
        df = df,
        rgb_pool=rgb_pool,
        savedir=tmpdir,
        raw_box_savedir=tmpdir
    ) 
    
    assert len(glob.glob("{}/*.shp".format(tmpdir))) > 0

def test_generate_crops(tmpdir):
    data_path = "{}/tests/data/crown.shp".format(ROOT)
    gdf = gpd.read_file(data_path)
    annotations = generate.generate_crops(
        gdf=gdf, rgb_glob="{}/tests/data/*.tif".format(ROOT),
        convert_h5=False, sensor_glob="{}/tests/data/*.tif".format(ROOT), savedir=tmpdir)
    
    assert not annotations.empty
    assert all([x in ["image_path","label","site","siteID","plotID","individualID","taxonID","point_id","box_id"] for x in annotations.columns])
    assert len(annotations.box_id.unique()) == annotations.shape[0]