#Crop Training to match extent of ground truth
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import box

sensor_path = "data/raw/20170218_UH_CASI_S4_NAD83.pix"
ground_truth_path = "data/raw/2018_IEEE_GRSS_DFC_GT_TR.tif"
destination_path = "data/processed/20170218_UH_CASI_S4_NAD83.tif"

ground_truth_src = rasterio.open(ground_truth_path)
bounds = ground_truth_src.bounds

#Read sensor data and crop
sensor_src = rasterio.open(sensor_path)
out_img, out_transform = mask(dataset=sensor_src, shapes=[box(*bounds)], crop=True)
#Select first 48 bands as channels
out_img = out_img[:48,:,:]

# Copy the metadata and write
out_meta = sensor_src.meta.copy()
out_meta.update({"driver": "GTiff",
                 "height": out_img.shape[1],
                 "width": out_img.shape[2],
                 "count": out_img.shape[0],
                 "transform": out_transform})

with rasterio.open(destination_path, "w", **out_meta) as dest:
    dest.write(out_img)
    
