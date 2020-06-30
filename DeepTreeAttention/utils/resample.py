#Resample raster to desired resolution
import rasterio
from rasterio.enums import Resampling

def resample(path, upscale_factor=2):
    upscale_factor = upscale_factor
    
    with rasterio.open(path) as dataset:
    
        # resample data to target shape
        data = dataset.read(
            out_shape=(
                dataset.count,
                int(dataset.height * upscale_factor),
                int(dataset.width * upscale_factor)
            ),
            resampling=Resampling.bilinear
        )
    
        # scale image transform
        transform = dataset.transform * dataset.transform.scale(
            (dataset.width / data.shape[-1]),
            (dataset.height / data.shape[-2])
        )
        
        #write new dataset
        metadata = src.meta.copy()
    
    metadata.update({
        'transform': transform,
    })
    
    basename = os.path.splitext(os.path.basename(path))[0]
    filename = "{}/{}_resampled.tif".format(savedir, basename)
    
    with rasterio.open(filename, "w", **metadata) as dest:
        data.write(img)
        
    new_dataset.write(arr)
    new_dataset.close()        