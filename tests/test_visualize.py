#test validate
from src import visualize, generate
from deepforest import main
import matplotlib.pyplot as plt
import rasterio
import geopandas as gpd

def test_crown_plot(rgb_path,plot_data):    
    m = main.deepforest()
    m.use_release(check_release=False)
    boxes = generate.predict_trees(deepforest_model=m, rgb_path=rgb_path, bounds=plot_data.total_bounds)
    src = rasterio.open(rgb_path)
    fig, ax = plt.subplots(figsize=(4, 4))
    rasterio.plot.show(src, ax=ax)
    boxes.plot(ax=ax, facecolor="none", edgecolor="red")
    #print(boxes.total_bounds[0])
    #plt.show()
    merged_boxes = gpd.sjoin(boxes, plot_data)    
    geom = merged_boxes.geometry.iloc[0]
    point = plot_data[plot_data.individual==merged_boxes.individual.iloc[0]].iloc[0].geometry
    visualize.crown_plot(rgb_path, geom=geom, point=point)
    print("Visualize crown plot completed")
    #plt.show()

def test_confusion_matrix(experiment, rgb_path, plot_data):
    m = main.deepforest()
    m.use_release(check_release=False)
    boxes = generate.predict_trees(deepforest_model=m, rgb_path=rgb_path, bounds=plot_data.total_bounds)    
    merged_boxes = gpd.sjoin(boxes, plot_data)    
    
    labels = plot_data.taxonID.unique()
    yhats = plot_data.taxonID.astype('category').cat.codes
    y = plot_data.taxonID.astype('category').cat.codes
    
    visualize.confusion_matrix(comet_experiment=experiment,
                               yhats=yhats,
                               y=y,
                               labels = labels, 
                               test=plot_data,
                               test_points=plot_data,
                               test_crowns=merged_boxes,
                               name="pytest")
    