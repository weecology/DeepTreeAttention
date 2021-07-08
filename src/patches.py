#Patches
import geopandas as gpd
from src import neon_paths
import rasterio

def row_col_from_crown(crown, src):
    """Given a geometry object and rasterio src, get the row col indices of all overlapping pixels
    Args:
        crown: shapely geometry box
        img_centroids: a list of (row, col) indices for the rasterio src object
    """
    left, bottom, right, top = crown.bounds 
    img_centroids = []
    img = src.read(window=rasterio.windows.from_bounds(left, bottom, right, top, transform=src.transform))
    win_transform = src.window_transform(window=rasterio.windows.from_bounds(left, bottom, right, top, transform=src.transform))
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            coords = rasterio.transform.xy(win_transform, row, col, offset='center')
            row_col = rasterio.transform.rowcol(src.transform, coords[0], coords[1])
            img_centroids.append(row_col)    
    
    return img_centroids
                    
def crown_to_pixel(crown, rgb_path, width=11, height=11):
    """Given a crown box, create the pixel crops"""
    crown_patches = []    
    src = rasterio.open(rgb_path)    
    img_centroids = row_col_from_crown(crown, src)
    for indices in img_centroids:
        row, col = indices
        img = src.read(window = rasterio.windows.Window(col_off=col, row_off=row, width = width, height=height), boundless=True)
        crown_patches.append(img)

    return crown_patches
