#test generator
import tensorflow as tf
import pandas as pd
import numpy as np
import rasterio
import os
from DeepTreeAttention.generators import generator

@pytest.fixture()
def create_image():
    # create fake image input (only shape is used anyway) # logic from https://github.com/fizyr/tf-retinanet/blob/master/tests/layers/test_misc.py
    current_folder = os.path.dirname(os.path.abspath(__file__))
    example_tif = os.path.join(current_folder,"data/example.tif")    
    d = rasterio.open(example_tif)
    src = d.read()
    
    with rasterio.Env():
    
        # Write an array as a raster band to a new 8-bit file. For
        # the new file's profile, we start with the profile of the source
        profile = src.profile
    
        # And then change the band count to 1, set the
        # dtype to uint8, and specify LZW compression.
        profile.update(
            dtype=rasterio.uint8,
            count=1,
            compress='lzw')
    
        with rasterio.open('example.tif', 'w', **profile) as dst:
            dst.write(array.astype(rasterio.uint8), 1)
            

    return image

#Create 10 images in a tempdir
@pytest.fixture(scope="session")
def image_file(tmpdir_factory):
    image_paths = []
    for i in np.arange(10):
        fn = tmpdir_factory.mktemp("data").join("img{}.tif")        
        img = create_image()
        img.save(str(fn))
        image_paths.append(fn)
        
    return image_paths