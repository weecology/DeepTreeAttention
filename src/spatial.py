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
        geom = gdf[gdf.index==x].geometry.centroid.buffer(buffer).iloc[0]
        touches = gdf[gdf.geometry.map(lambda x: x.intersects(geom))]
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
    
    