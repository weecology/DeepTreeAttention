#spatial neighbors
import numpy as np
import geopandas as gpd
from src.patches import crop
from src.neon_paths import find_sensor_path
from src.data import preprocess_image
import torch
from torchvision import transforms

def spatial_neighbors(gdf, buffer, data_dir, HSI_pool, model, image_size):
    """    
    #Get all neighbors within n meters of each point.
    Args:
        gdf: a geodataframe
        buffer: distance from focal point in m to search for neighbors
        data_dir: directory where the plot boxes are stored
        HSI_pool: path to search for sensor path
        model: a trained main.TreeModel to predict neighbor box scores
        image_size: 
    Returns:
        neighbors: dictionary with keys -> index of the gdf, value of index of neighbors
    """
    model.model.eval()
    neighbors = {}
    for x in gdf.index:
        print(x)
        geom = gdf[gdf.index==x].geometry.centroid.buffer(buffer).iloc[0]
        plotID = gdf.plotID.unique()[0]   
        #Read existing box
        neighbor_boxes = gpd.read_file("{}/interim/{}_boxes.shp".format(data_dir, plotID))
        #Finding crowns that are within buffer distance
        touches = neighbor_boxes[neighbor_boxes.geometry.map(lambda x: x.intersects(geom))]
        scores = []
        for b in touches.geometry:
            #Predict score
            print(b.bounds)
            print(sensor_path)
            try:
                sensor_path = find_sensor_path(lookup_pool=HSI_pool, bounds=b.bounds)                
                img_crop = crop(bounds=b.bounds, sensor_path=sensor_path)
            except Exception as e:
                print(e)
                continue
            img_crop = preprocess_image(img_crop, channel_is_first=True)
            img_crop = transforms.functional.resize(img_crop, size=(image_size,image_size), interpolation=transforms.InterpolationMode.NEAREST)
            img_crop = torch.tensor(img_crop,device=model.device, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                score = model.model(img_crop)
            scores.append(score)
        
        if len(scores) == 0:
            neighbors[x] = torch.zeros(1, model.classes, device=model.device, dtype=torch.float32).unsqueeze(0)
        else:            
            neighbors[x] = np.vstack(scores)
    
    print(neighbors)
    return neighbors
    
def spatial_smooth(neighbors, features, alpha=0.2):
    """Average the predictions spatially to create a neighborhood effect
    Args:
        neighbors: a geodataframe resulting from main.predict_dataloader()
        features: matrix of features to smooth, in the same order as the test dataset
    Returns:
        labels: predicted taxa after spatial function
        scores: confidence score for predicted spatial taxa
    """
    smoothed_features = []
    for x in range(features.shape[0]):
        spatial_features = neighbors[x]
        spatial_features = np.mean(spatial_features, axis=0)
        focal_features = features[x,]
        
        #if no neighbors, return itself
        if spatial_features.size == 0:
            smoothed_features.append(focal_features)
        else:
            smoothed_feature = focal_features + (alpha * spatial_features)
            smoothed_features.append(smoothed_feature)
        
    smoothed_features = np.vstack(smoothed_features)
    labels = np.argmax(smoothed_features, 1)
    scores = smoothed_features[np.arange(len(labels)),labels]
    
    return labels, scores
    
    