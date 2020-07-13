#Resample raster to desired resolution
import rasterio
import os
from rasterio.enums import Resampling

def create_tif(source_tif, filename, numpy_array):
    """Create a tif file with metadata from numpy array"""
    
    with rasterio.open(source_tif) as src:
        #write new dataset
        metadata = src.meta.copy()  
    
    metadata.update({
        'count': 1,
    })    
    
    with rasterio.open(filename, "w", **metadata) as dest:
        dest.write(numpy_array)    
    
def resample(path, upscale_factor=2):    
    """Resample resolution of .tif and return filename"""
    with rasterio.open(path) as dataset:
    
        # resample data to target shape
        data = dataset.read(
            out_shape=(
                dataset.count,
                int(dataset.height * upscale_factor),
                int(dataset.width * upscale_factor)
            ),
            resampling=Resampling.nearest
        )
    
        # scale image transform
        transform = dataset.transform * dataset.transform.scale(
            (dataset.width / data.shape[-1]),
            (dataset.height / data.shape[-2])
        )
        
        #write new dataset
        metadata = dataset.meta.copy()
    
    metadata.update({
        'transform': transform,
        "height": dataset.height*upscale_factor,
        "width":dataset.width*upscale_factor
    })
    
    basename = os.path.splitext(path)[0]
    filename = "{}_resampled.tif".format(basename)
    
    with rasterio.open(filename, "w", **metadata) as dest:
        dest.write(data)
            
    return filename