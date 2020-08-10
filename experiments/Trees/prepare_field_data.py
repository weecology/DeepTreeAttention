#Convert NEON field sample points into bounding boxes of cropped image data for model training
import cv2

from DeepTreeAttention.generators.boxes import write_tfrecord

#What to do about classes?

def find_sensor_data(plot_name, sensor="hyperspectral"):
    pass

def resize(img, height, width):
    # resize image
    dim = (width, height)    
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
    return resized

def process_plot(plot_data):
    """For a given NEON plot, find the correct sensor data, predict trees and associate bounding boxes with field data
    Args:
        plot_data: geopandas dataframe in a utm projection
        
    """
    #DeepForest prediction
    plot_name = plot_data.plotID.unique()[0]
    if len(plot_name) > 1:
        raise ValueError("Multiple plots passed to plot_data argument")
    
    rgb_sensor_path = find_sensor_data(plot_name, sensor="rgb")
    boxes = predict_trees(rgb_sensor_path)

    #Merge results with field data
    merged_boxes = merge_points_boxes(plot_data, boxes)

    #Remove unclassified
    merged_boxes = merged_boxes[~(merged_boxes.label==0)]
    
    return merged_boxes

def create_crops():
    """Crop sensor data based on a dataframe of geopandas bounding boxes"""
    crops = []
    labels = []
    box_index = []
    for index, row in merged_boxes.iterrows():
        box = row["geometry"]       
        plot_name = row["plotID"]                
        sensor_path = find_sensor_data(box, sensor="hyperspectral")        
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
        resized_crops = [resize(x, height, width).astype("int16") for x in chunk_crops]
        
        filename = "{}/{}_{}.tfrecord".format(savedir, basename, counter)
        write_tfrecord(filename=filename,
                                            images=resized_crops,
                                            labels=chunk_labels,
                                            indices=chunk_index,
                                            classes=classes)
        
        filenames.append(filename)
        counter +=1    
    
def main(field_data, savedir=".", chunk_size=1000):
    """Prepare NEON field data into tfrecords
    Args:
        field_data: csv file with location and class of each field collected point
        savedir: direcory to save completed tfrecords
    """
    neon_data = read_data()
    plot_names = get_plotnames()
    
    merged_boxes = []
    for plot in plot_names:
        #Filter data
        plot_data = neon_data[neon_data.plotID == plot]
        predicted_trees = process_plot(plot_data)
        merged_boxes.append(predicted_trees)
        
    #Get sensor data
    crops, labels, box_index = create_crops(merged_boxes)
        
    #Write tfrecords
    create_records(crops, labels, box_index, savedir, chunk_size=chunk_size)
    
if __name__ == "__main__":
    main()