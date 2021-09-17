#Patches
import rasterio

def crop(bounds, sensor_path, savedir = None, basename = None):
    """Given a 4 pointed bounding box, crop sensor data"""
    left, bottom, right, top = bounds 
    height = top - bottom
    width = right - left
    src = rasterio.open(sensor_path)        
    img = src.read(window=rasterio.windows.from_bounds(left, bottom, right, top, transform=src.transform))    
    if savedir:
        filename = "{}/{}.tif".format(savedir, basename)
        with rasterio.open(filename, "w", driver="GTiff",height=height, width=width, count = img.shape[0], dtype=img.dtype) as dst:
            dst.write(img)
    if savedir:
        return filename
    else:
        return img    
    
def row_col_from_bounds(bounds, src):
    """Given a geometry object and rasterio src, get the row col indices of all overlapping pixels
    Args:
        bounds: bounds of geometry object or raster tile
        img_centroids: a list of (row, col) indices for the rasterio src object
    """
    left, bottom, right, top = bounds 
    img_centroids = []
    img = src.read(window=rasterio.windows.from_bounds(left, bottom, right, top, transform=src.transform))
    win_transform = src.window_transform(window=rasterio.windows.from_bounds(left, bottom, right, top, transform=src.transform))
    for row in range(img.shape[1]):
        for col in range(img.shape[2]):
            coords = rasterio.transform.xy(win_transform, row, col, offset='center')
            row_col = rasterio.transform.rowcol(src.transform, coords[0], coords[1])
            img_centroids.append(row_col)    
    
    return img_centroids
                    
def bounds_to_pixel(bounds, img_path, savedir=None, basename=None,width=11, height=11):
    """Given a crown box, create the pixel crops
    Args:
         crown: a geometry object
         img_path: sensor data to crop
         savedir: location to save crops, if none, crops are returned
         basename: output file is {basename}_{counter}.tif for each pixel crop
         width: pixel size crop in x
         height: pixel size in y
    Returns:
         crops: [(row, col), image crop]
         filenames: filenames of written patches
    """
    counter = 0
    filenames = []   
    crops = []
    src = rasterio.open(img_path)    
    img_centroids = row_col_from_bounds(bounds, src)
    for indices in img_centroids:
        row, col = indices
        img = src.read(window = rasterio.windows.Window(col_off=col, row_off=row, width = width, height=height), boundless=True)
        if savedir:
            filename = "{}/{}_{}.tif".format(savedir, basename, counter)
            with rasterio.open(filename, "w", driver="GTiff",height=height, width=width, count = img.shape[0], dtype=img.dtype) as dst:
                dst.write(img)
            counter = counter + 1
            filenames.append(filename)

        else:
            crops.append([(row,col),img])
    if savedir:
        return filenames
    else:
        return crops
        
