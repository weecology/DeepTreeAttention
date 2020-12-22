#Context module. Use a pretrain model to extract the penultimate layer of the model for surrounding trees.
import tensorflow as tf
import rasterio

from DeepTreeAttention.generators.boxes import crop_image, resize
from DeepTreeAttention.utils.paths import find_sensor_path, elevation_from_tile
from DeepTreeAttention.generators import neighbors

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

def extract_features(df, x, model, hyperspectral_pool, site_label_dict, domain_label_dict, HSI_size=20, k_neighbors=5):
    """Generate features
    Args:
    df: a geopandas dataframe
    x: individual id to use a target
    model: A deeptreeattention model class to extract layer features
    hyperspectral_pool: glob dir to search for sensor files
    HSI_size: size of HSI crop
    site_label_dict: dictionary of numeric site labels
    domain_label_dict: dictionary of numeric domain labels
    k_neighbors: number of neighbors to extract
    Returns:
    feature_array: a feature matrix of encoded bottleneck layer
    """
    #Due to resampling, there will be multiple rows of the same point, all are identical.
    target  =  df[df.individual == x].head(1)
    sensor_path = find_sensor_path(bounds=target.total_bounds, lookup_pool=hyperspectral_pool) 
    
    #Encode metadata
    site = target.siteID.values[0]
    numeric_site = site_label_dict[site]
    one_hot_sites = tf.one_hot(numeric_site, model.sites)
    
    domain = target.siteID.values[0]
    numeric_domain = domain_label_dict[domain]   
    one_hot_domains = tf.one_hot(numeric_domain, model.domains)
    
    elevation = elevation_from_tile(sensor_path)/1000
    metadata = [elevation, one_hot_sites, one_hot_domains]
    
    neighbor_pool = df[~(df.individual == x)]
    raster = rasterio.open(sensor_path)
    feature_array = predict_neighbors(target, metadata=metadata, HSI_size=HSI_size, raster=raster, neighbor_pool=neighbor_pool, model=model.ensemble_model, k_neighbors=k_neighbors)
    
    return feature_array

def predict_neighbors(target, HSI_size, neighbor_pool, metadata, raster, model, k_neighbors=5):
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
    neighbor_geoms = get_nearest(target, candidates = neighbor_pool , k_neighbors=k_neighbors)
    
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
    
        
    