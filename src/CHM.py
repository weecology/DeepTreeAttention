#CHM height module. Given a x,y location and a pool of CHM images, find the matching location and extract the crown level CHM measurement
import glob
import numpy as np 
from src import neon_paths
import rasterstats
import geopandas as gpd
import pandas as pd

def non_zero_99_quantile(x):
    """Get height quantile of all cells that are no zero"""
    mdata = np.ma.masked_where(x < 0.5, x)
    mdata = np.ma.filled(mdata, np.nan)
    percentile = np.nanpercentile(mdata, 99)
    return (percentile)

def postprocess_CHM(df, lookup_pool):
    """Field measured height must be within min_diff meters of canopy model"""
    #Extract zonal stats
    try:
        CHM_path = neon_paths.find_sensor_path(lookup_pool=lookup_pool, bounds=df.total_bounds)
    except Exception as e:
        raise ValueError("Cannot find CHM path for {} from plot {} in lookup_pool: {}".format(df.total_bounds, df.plotID.unique(),e))
    draped_boxes = rasterstats.zonal_stats(df.geometry.__geo_interface__,
                                           CHM_path,
                                           add_stats={'q99': non_zero_99_quantile})
    df["CHM_height"] = [x["q99"] for x in draped_boxes]

    #if height is null, assign it
    df.height.fillna(df["CHM_height"], inplace=True)
        
    return df
        
def CHM_height(shp, CHM_pool):
        """For each plotID extract the heights from LiDAR derived CHM
        Args:
            shp: shapefile of data to filter
            config: DeepTreeAttention config file dict, parsed, see config.yml
        """    
        filtered_results = []
        lookup_pool = glob.glob(CHM_pool, recursive=True)        
        for name, group in shp.groupby("plotID"):
            try:
                result = postprocess_CHM(group, lookup_pool=lookup_pool)
                filtered_results.append(result)
            except Exception as e:
                print("plotID {} raised: {}".format(name,e))
                
        filtered_shp = gpd.GeoDataFrame(pd.concat(filtered_results,ignore_index=True))
        
        return filtered_shp
    
def filter_CHM(shp, CHM_pool, min_CHM_height=1, min_CHM_diff=4):
    
    if min_CHM_height is None:
        return shp
    
    #extract CHM height
    shp = CHM_height(shp, CHM_pool)
    
    #Remove NULL CHM_heights
    #shp = shp[~(shp.CHM_height.isnull())]
    
    shp = shp[(shp.height.isnull()) | (shp.CHM_height > min_CHM_height)]
    
    #remove CHM points under height diff  
    shp = shp[(shp.height.isnull()) | (abs(shp.height - shp.CHM_height) < min_CHM_diff)]  
    
    return shp