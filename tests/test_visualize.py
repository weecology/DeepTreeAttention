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

def test_confusion_matrix(experiment, rgb_path, plot_data, tmpdir, ROOT):
    m = main.deepforest()
    m.use_release(check_release=False)
    gdf = generate.predict_trees(deepforest_model=m, rgb_path=rgb_path, bounds=plot_data.total_bounds)  
    gdf["individual"] = gdf.index.values
    
    img_pool = glob.glob("{}/tests/data/*.tif".format(ROOT))
    
    annotations = generate.generate_crops(
        gdf=gdf,
        rgb_pool=img_pool,
        convert_h5=False,
        img_pool=img_pool,
        h5_pool=img_pool,
        savedir=tmpdir
    )
    
    rgb_crowns = gdf.copy()
    rgb_crowns.geometry = rgb_crowns.geometry.buffer(1)
    rgb_annotations = generate.generate_crops(
        rgb_crowns,
        savedir=tmpdir,
        img_pool=img_pool,
        h5_pool=None,
        rgb_pool=img_pool,
        convert_h5=False,   
        client=None,
        suffix="RGB"
    )
    rgb_annotations["RGB_image_path"] = rgb_annotations["image_path"]
    annotations["tile_year"] = rgb_annotations["tile_year"]
    annotations = annotations.merge(rgb_annotations[["individual","tile_year","RGB_image_path"]], on=["individual","tile_year"])
    annotations["taxonID"] = "A"
    
    labels = annotations.taxonID.unique()
    yhats = annotations.taxonID.astype('category').cat.codes.values
    y = annotations.taxonID.astype('category').cat.codes.values
    