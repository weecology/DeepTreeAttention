#test validate
from src import visualize, generate
from deepforest import main
import matplotlib.pyplot as plt
import rasterio
import geopandas as gpd
import glob

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

def test_confusion_matrix(experiment, dm):    
    visualize.confusion_matrix(
        comet_experiment=experiment,
        yhats=dm.test.label.values,
        y=dm.test.label.values,
        labels=list(dm.species_label_dict.keys()),
        test=dm.test,
        test_points=dm.canopy_points,
        test_crowns=dm.crowns,
        crop_dir=dm.config["crop_dir"],
        name="pytest"
    )    