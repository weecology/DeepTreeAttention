#spatial neighbors
import numpy as np
import geopandas as gpd
from src.patches import crop
from src.neon_paths import find_sensor_path
import torch

def spatial_neighbors(gdf, buffer, data_dir, rgb_pool, model):
    """    
    #Get all neighbors within n meters of each point.
    Args:
        gdf: a geodataframe
        buffer: distance from focal point in m to search for neighbors
        data_dir: directory where the plot boxes are stored
        model: a trained main.TreeModel to predict neighbor box scores
    Returns:
        neighbors: dictionary with keys -> index of the gdf, value of index of neighbors
    """
    model.eval()
    neighbors = {}
    for x in gdf.index:
        geom = gdf[gdf.index==x].geometry.centroid.buffer(buffer).iloc[0]
        plotID = gdf.plotID.unique()[0]   
        #Read existing box
        neighbor_boxes = gpd.read_file("{}/interim/{}_boxes.shp".format(data_dir, plotID))
        #Finding crowns that are within buffer distance
        touches = neighbor_boxes[neighbor_boxes.geometry.map(lambda x: x.intersects(geom))]
        boxes = [i for i in touches.index if not x == i]
        rgb_path = find_sensor_path(lookup_pool=rgb_pool, bounds=geom.bounds)
        scores = []
        for b in boxes.geometry:
            #Predict score
            img_crop = crop(bounds=b, sensor_path=rgb_path)
            img_crop = torch.tensor(img_crop,device=model.device).unsqueeze(0)
            score = model(img_crop)
            scores.append(score)
            
        neighbors[x] = scores

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
        spatial_features = [features[i,] for i in neighbors[x]]
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
    
    