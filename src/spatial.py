#spatial neighbors
import numpy as np

def spatial_neighbors(gdf, buffer):
    """    
    #Get all neighbors within n meters of each point.
    Args:
        gdf: a geodataframe
        buffer: distance from focal point in m to search for neighbors
    Returns:
        neighbors: dictionary with keys -> index of the gdf, value of index of neighbors
    """
    neighbors = {}
    for x in gdf.index:
        geom = gdf.loc[x].geometry.centroid.buffer(buffer)
        touches = gdf.intersection(geom)
        touches = touches[~(touches.geometry.is_empty)]
        neighbors[x] = [i for i in touches.index if not x == i]
    
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
    for x in neighbors:
        neighbor_index = neighbors[x]
        focal_features = features[x,:]
        spatial_features = features[neighbor_index,:]
        #if no neighbors, return itself
        if spatial_features.size == 0:
            smoothed_features.append(focal_features)
        else:
            smoothed_feature = focal_features + (alpha * np.mean(spatial_features))
            smoothed_features.append(smoothed_feature)
        
    smoothed_features = np.vstack(smoothed_features)
    labels = np.argmax(smoothed_features, 1)
    scores = smoothed_features[np.arange(len(labels)),labels]
    
    return labels, scores
    
    