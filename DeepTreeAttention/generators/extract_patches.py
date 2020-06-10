##Generate patches from a large raster##
"""preprocessing model for creating a non-overlapping sliding window of fixed size to generate tfrecords for model training"""
import rasterio

def _read_file(path):
    """Read a hyperspetral raster .tif 
    Args:
        path: a path to a .tif hyperspectral raster
    Returns:
        src: a numpy array of height x width x channels
        """
    r = rasterio.open()
    src = r.read()
    
    return src
    
def run():
    raster = _read_file()