import geopandas as gpd
import glob
import numpy as np
import pandas as pd
import rasterstats

from shapely.geometry import Point
from DeepTreeAttention.utils.paths import find_sensor_path


def non_zero_99_quantile(x):
    """Get height quantile of all cells that are no zero"""
    mdata = np.ma.masked_where(x < 0.5, x)
    mdata = np.ma.filled(mdata, np.nan)
    percentile = np.nanpercentile(mdata, 99)
    return (percentile)

def postprocess_CHM(df, lookup_pool, min_diff=4):
    """Field measured height must be within min_diff meters of canopy model"""
    #Extract zonal stats
    CHM_path = find_sensor_path(lookup_pool=lookup_pool, bounds=df.total_bounds)
    draped_boxes = rasterstats.zonal_stats(df.geometry.__geo_interface__,
                                           CHM_path,
                                           add_stats={'q99': non_zero_99_quantile})
    df["CHM_height"] = [x["q99"] for x in draped_boxes]

    #Rename column
    df = df[abs(df.height - df.CHM_height) < min_diff]

    return df

#Load test data and create shapefile
def test_split(path):
    ids = pd.read_csv(path)
    ids = ids[["itcEasting","itcNorthing","siteID", "plotID", "elevation","domainID","individualID","taxonID"]]   
    
    #One invalid tile species
    ids = ids[~(ids.taxonID == "GYDI")]
    
    ids["geometry"] = [Point(x,y) for x,y in zip(ids["itcEasting"], ids["itcNorthing"])]
    shp = gpd.GeoDataFrame(ids)
    
    return shp

def train_split(path, test_ids, test_species, debug = False):
    """Create a train split from a larger pool of data, excluding any test ids"""
    field = pd.read_csv(path)
    
    if debug:
        field = field.sample(n=2000)
        
    #Inclusion criteria 
    train_field = field[~(field.individualID.isin(test_ids))]
    has_elevation = train_field[~train_field.elevation.isnull()]
    alive = has_elevation[has_elevation.plantStatus=="Live"]
    trees = alive[~alive.growthForm.isin(["liana","small shrub"])]
    trees = trees[~trees.growthForm.isnull()]
    latest_year = trees.groupby("individualID").apply(lambda x: x.sort_values(["eventID"]).head(1))
    sun_position = latest_year[~(latest_year.canopyPosition.isin(["Full shade", "Mostly shaded"]))]
    min_height = sun_position[(sun_position.height > 3) | (sun_position.height.isnull())]
    min_size = min_height[min_height.stemDiameter > 5]
    min_date = min_size[~(min_size.eventID.str.contains("2014"))]
    
    #drop one species on invalid data
    min_date = min_date[~(min_date.taxonID == "GYDI")]
    
    #ensure that species set matches
    min_date = min_date[min_date.taxonID.isin(test_species)]
    
    #Create shapefile
    min_date["geometry"] = [Point(x,y) for x,y in zip(min_date["itcEasting"], min_date["itcNorthing"])]
    shp = gpd.GeoDataFrame(min_date)
    
    shp = shp[["siteID","plotID","height","elevation","domainID","individualID","taxonID","itcEasting","itcNorthing","geometry"]]
    
    return shp
        
def filter_CHM(train_shp, lookup_glob):
        """For each plotID extract the heights from LiDAR derived CHM"""    
        filtered_results = []
        lookup_pool = glob.glob(lookup_glob)        
        for name, group in train_shp.groupby("plotID"):
            result = postprocess_CHM(group, lookup_pool=lookup_pool, min_diff=4)
            filtered_results.append(result)
        filtered_shp = gpd.GeoDataFrame(pd.concat(filtered_results,ignore_index=True))
        
        return filtered_shp