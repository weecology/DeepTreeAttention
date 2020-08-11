#Convert NEON field sample points into bounding boxes of cropped image data for model training
import cv2
import os
import glob
import geopandas as gpd

from DeepTreeAttention.generators.boxes import write_tfrecord
from deepforest import deepforest

def resize(img, height, width):
    # resize image
    dim = (width, height)    
    img = img.astype("float32")
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_NEAREST)
    
    return resized

def predict_trees(deepforest_model, rgb_path):
    boxes = deepforest_model.predict_tile(rgb_path)
    return boxes

def process_plot(plot_data, rgb_pool, deepforest_model):
    """For a given NEON plot, find the correct sensor data, predict trees and associate bounding boxes with field data
    Args:
        plot_data: geopandas dataframe in a utm projection
    Returns:
        merged_boxes: geodataframe of bounding box predictions with species labels
    """
    #DeepForest prediction
    
    rgb_sensor_path = find_sensor_path(bounds=plot_data.total_bounds,lookup_pool=rgb_pool)
    boxes = predict_trees(rgb_sensor_path)

    #Merge results with field data
    merged_boxes = merge_points_boxes(plot_data, boxes)

    #Remove unclassified
    merged_boxes = merged_boxes[~(merged_boxes.label==0)]
    
    return merged_boxes

def create_crops(merged_boxes, hyperspectral_pool, sensor="hyperspectral"):
    """Crop sensor data based on a dataframe of geopandas bounding boxes
    Args:
        merged_boxes: geopandas dataframe with bounding box geometry, plotID, and species label
        hyperspectral_pool: glob string for looking up matching sensor tiles
    Returns:
        crops: list of cropped sensor data
        labels: species id labels
        box_index: unique index and plot_data length.
    """
    crops = []
    labels = []
    box_index = []
    for index, row in merged_boxes.iterrows():
        box = row["geometry"]       
        plot_name = row["plotID"]                
        sensor_path = find_sensor_path(bounds=box.bounds, lookup_pool=hyperspectral_pool, sensor=sensor)        
        crop = crop_image(sensor_path, box)
        labels.append(row["label"])
        box_index.append("{}_{}".format(plot_name,index))
        
    return crops, labels, box_index

def create_records(crops, labels, box_index, savedir, chunk_size=1000):
    #get keys and divide into chunks for a single tfrecord
    filenames = []
    counter = 0
    for i in range(0, len(crops)+1, chunk_size):
        chunk_crops = crops[i:i + chunk_size]
        chunk_index = indices[i:i + chunk_size]
        
        if train:
            chunk_labels = labels[i:i + chunk_size]
        else:
            chunk_labels = None
        
        #resize crops
        resized_crops = [resize(x, height, width) for x in chunk_crops]
        
        filename = "{}/{}_{}.tfrecord".format(savedir, basename, counter)
        write_tfrecord(filename=filename,
                                            images=resized_crops,
                                            labels=chunk_labels,
                                            indices=chunk_index,
                                            classes=classes)
        
        filenames.append(filename)
        counter +=1    
    
def main(field_data, rgb_pool=None, hyperspectral_pool=None, sensor="hyperspectral", savedir=".", chunk_size=1000):
    """Prepare NEON field data into tfrecords
    Args:
        field_data: shp file with location and class of each field collected point
        sensor: 'rgb' or 'hyperspecral' image crop
        savedir: direcory to save completed tfrecords
    """
    #Check sensor type has paths
    if sensor == "hyperspectral":
        assert not hyperspectral_pool is None
    if sensor=="rgb":
        assert not rgb_pool is None
        
    df = gpd.read_file(field_data)
    plot_names = df.plotID.unique()
    
    #create deepforest model
    deepforest_model = deepforest.deepforest()
    deepforest_model.use_release()
    
    merged_boxes = []
    for plot in plot_names:
        #Filter data
        plot_data = df[df.plotID == plot]
        predicted_trees = process_plot(plot_data, rgb_pool, deepforest_model)
        merged_boxes.append(predicted_trees)
        
    #Get sensor data
    crops, labels, box_index = create_crops(merged_boxes, lookup_pool=hyperspectral_pool, sensor=sensor)
        
    #Write tfrecords
    create_records(crops, labels, box_index, savedir, chunk_size=chunk_size)
    
if __name__ == "__main__":
    main("data/processed/field_data.shp")