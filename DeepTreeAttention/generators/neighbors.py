#Context module. Use a pretrain model to extract the penultimate layer of the model for surrounding trees.
import tensorflow.keras as tfk
from DeepTreeAttention.generators.boxes import crop_image, resize


from sklearn.neighbors import BallTree
import numpy as np

def get_nearest(src_points, candidates, k_neighbors=1):
    """Find nearest neighbors for all source points from a set of candidate points"""

    # Create tree from the candidate points
    coordinates = np.vstack(candidates.geometry.centroid.apply(lambda geom: (geom.x,geom.y)))
    tree = BallTree(coordinates, leaf_size=15, metric='haversine')

    # Find closest points and distances
    src_x = src_points.geometry.centroid.x
    src_y = src_points.geometry.centroid.y
    
    src_points = np.array([src_x, src_y]).reshape(-1,2)
    distances, indices = tree.query(src_points, k=k_neighbors)
    
    neighbor_geoms = candidates[candidates.index.isin(indices[0])]
    neighbor_geoms["distance"] = distances[0]

    # Return indices and distances
    return neighbor_geoms

def neighbors(target, HSI_size, neighbor_pool, metadata, raster, model, n=5):
    """Get features of surrounding n trees
    Args:
    target: geometry object of the target point
    neighbor_pool: geopandas dataframe with points
    metadata: The metadata layer for each of the points, assumed to be identical for all neighbors
    n: Number of neighbors
    model: A model object to predict features
    Returns:
    n * m feature matrix, where n is number of neighbors and m is length of the penultimate model layer
    """
        
    #Find neighbors
    neighbor_geoms = get_nearest(target, candidates = neighbor_pool , k_neighbors=n)
    
    #extract crop for each neighbor
    features = [ ]
    for neighbor in neighbor_geoms.geometry:
        crop = crop_image(src=raster, box=neighbor)
        crop = resize(crop, HSI_size, HSI_size)
        crop = np.expand_dims(crop, 0)
        #create batch
        batch  = [crop,metadata[0],metadata[1],metadata[2]]
        feature = model(batch)
        features.append(feature)
    
    features = np.vstack(features)
    
    return features
    
        
    