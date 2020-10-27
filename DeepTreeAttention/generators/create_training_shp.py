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

def postprocess_CHM(df, lookup_pool, min_diff=1, remove=True):
    """Field measured height must be within min_diff meters of canopy model"""
    #Extract zonal stats
    CHM_path = find_sensor_path(lookup_pool=lookup_pool, bounds=df.total_bounds)
    draped_boxes = rasterstats.zonal_stats(df.geometry.__geo_interface__,
                                           CHM_path,
                                           add_stats={'q99': non_zero_99_quantile})
    df["CHM_height"] = [x["q99"] for x in draped_boxes]

    #if height is null, assign it
    df.height.fillna(df["CHM_height"], inplace=True)
    
    #Rename column
    if remove:
        df = df[(abs(df.height - df.CHM_height) < min_diff)]

    return df

#Load test data and create shapefile
def test_split(path, field_data_path):
    ids = pd.read_csv(path)
    field_data = pd.read_csv(field_data_path)
    ids = ids[["itcEasting","itcNorthing","siteID", "plotID", "elevation","domainID","individualID","taxonID"]]   
    
    field_data = field_data[field_data.individualID.isin(ids.individualID.unique())]
    merge_height = field_data.groupby("individualID").apply(lambda x: x.sort_values(["eventID"],ascending=False).head(1)).reset_index(drop=True)
    merge_height = merge_height[["individualID","height"]]

    ids = ids.merge(merge_height)
    
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
    latest_year = trees.groupby("individualID").apply(lambda x: x.sort_values(["eventID"],ascending=False).head(1))
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
    
    #drop and reset
    shp = shp.drop(columns="individualID").reset_index()
    
    return shp
        
def filter_CHM(train_shp, lookup_glob, remove=True):
        """For each plotID extract the heights from LiDAR derived CHM"""    
        filtered_results = []
        lookup_pool = glob.glob(lookup_glob, recursive=True)        
        for name, group in train_shp.groupby("plotID"):
            try:
                result = postprocess_CHM(group, lookup_pool=lookup_pool, min_diff=4, remove = remove)
            except Exception as e:
                print("plotID: {} failed with {}".format(group.plotID.unique(),e))
                continue
            filtered_results.append(result)
        filtered_shp = gpd.GeoDataFrame(pd.concat(filtered_results,ignore_index=True))
        
        return filtered_shp
    
def train_test_split(ROOT, lookup_glob):
    """Create the train test split
    Args:
        ROOT: 
        lookup_glob: The recursive glob path for the canopy height models to create a pool of .tif to search
        """
    test = test_split("{}/data/raw/test_with_uid.csv".format(ROOT), field_data_path="{}/data/raw/latest_full_veg_structure.csv".format(ROOT))
    #Interpolate CHM height
    test = filter_CHM(test, lookup_glob, remove=False)
    train = train_split("{}/data/raw/latest_full_veg_structure.csv".format(ROOT), test.individualID, test.taxonID.unique())
    
    #write sample test data
    sample_data = train[train.plotID=="HARV_026"]
    sample_data.to_file("{}/experiments/Trees/test_data/sample.shp".format(ROOT))
    
    filtered_train = filter_CHM(train, lookup_glob, remove=True)
    filtered_train = filtered_train[filtered_train.taxonID.isin(test.taxonID.unique())]
    test = test[test.taxonID.isin(filtered_train.taxonID.unique())]

    print("There are {} records for {} species for {} sites in filtered train".format(
        filtered_train.shape[0],
        len(filtered_train.taxonID.unique()),
        len(filtered_train.siteID.unique())
    ))
    
    print("There are {} records for {} species for {} sites in test".format(
        test.shape[0],
        len(test.taxonID.unique()),
        len(test.siteID.unique())
    ))
    
    
    #just to be safe, assert no test in train
    check_empty = test[test.individualID.isin(train.individualID.unique())]
    assert check_empty.empty
    
    test.to_file("{}/data/processed/test.shp".format(ROOT))
    train.to_file("{}/data/processed/train.shp".format(ROOT))    
    filtered_train.to_file("{}/data/processed/CHM_filtered_train.shp".format(ROOT))    