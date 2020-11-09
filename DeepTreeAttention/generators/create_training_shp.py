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

def postprocess_CHM(df, lookup_pool):
    """Field measured height must be within min_diff meters of canopy model"""
    #Extract zonal stats
    try:
        CHM_path = find_sensor_path(lookup_pool=lookup_pool, bounds=df.total_bounds)
    except Exception as e:
        raise ValueError("Cannot find CHM path for {} from plot {} in lookup_pool: {}".format(df.total_bounds, df.plotID.unique(),e))
    draped_boxes = rasterstats.zonal_stats(df.geometry.__geo_interface__,
                                           CHM_path,
                                           add_stats={'q99': non_zero_99_quantile})
    df["CHM_height"] = [x["q99"] for x in draped_boxes]

    #if height is null, assign it
    df.height.fillna(df["CHM_height"], inplace=True)
        
    ##Rename column
    #if remove:
        ##drop points with less than 1 m height        
        #df = df[(abs(df.height - df.CHM_height) < min_diff)]

    return df

#Load test data and create shapefile
def test_split(path, field_data_path):
    ids = pd.read_csv(path)
    field_data = pd.read_csv(field_data_path)
    
    ids = ids[ids.plantStatus.str.contains("Live")]    
    ids = ids[["itcEasting","itcNorthing","siteID", "plotID", "elevation","plantStatus","domainID","individualID","taxonID"]]   
    
    field_data = field_data[field_data.individualID.isin(ids.individualID.unique())]
    merge_height = field_data.groupby("individualID").apply(lambda x: x.sort_values(["eventID"],ascending=False).head(1)).reset_index(drop=True)
    merge_height = merge_height[["individualID","height"]]

    ids = ids.merge(merge_height)
    
    # invalid tile species and plots
    ids = ids[~(ids.plotID == "KONZ_049")]
    ids = ids[~(ids.individualID == "NEON.PLA.D17.SOAP.03458")]
    
    ids["geometry"] = [Point(x,y) for x,y in zip(ids["itcEasting"], ids["itcNorthing"])]
    shp = gpd.GeoDataFrame(ids)
    
    return shp

def train_split(path, test_ids, test_species, debug = False):
    """Create a train split from a larger pool of data, excluding any test ids"""
    field = pd.read_csv(path)
    
    if debug:
        #field = field.sample(n=2000)
        field = field[field.siteID.isin(["BLAN","STEI"])]
        
    #Inclusion criteria 
    train_field = field[~(field.individualID.isin(test_ids))]
    has_elevation = train_field[~train_field.elevation.isnull()]
    trees = has_elevation[~has_elevation.growthForm.isin(["liana","small shrub"])]
    trees = trees[~trees.growthForm.isnull()]
    trees = trees[~trees.plantStatus.isnull()]        
    trees = trees[trees.plantStatus.str.contains("Live")]    
    sun_position = trees[~(trees.canopyPosition.isin(["Full shade", "Mostly shaded"]))]
    min_height = sun_position[(sun_position.height > 3) | (sun_position.height.isnull())]
    min_size = min_height[min_height.stemDiameter > 5]
        
    #ensure that species set matches
    min_size = min_size[min_size.taxonID.isin(test_species)]
    min_date = min_size[~(min_size.eventID.str.contains("2014"))]
    
    latest_year = min_date.groupby("individualID").apply(lambda x: x.sort_values(["eventID"],ascending=False).head(1))
    
    #Create shapefile
    latest_year["geometry"] = [Point(x,y) for x,y in zip(latest_year["itcEasting"], latest_year["itcNorthing"])]
    shp = gpd.GeoDataFrame(latest_year)
    
    #HOTFIX, BLAN has some data in 18N UTM, reproject to 17N update columns
    BLAN_errors = shp[(shp.siteID == "BLAN") & (shp.utmZone == "18N")]
    BLAN_errors.set_crs(epsg=32618, inplace=True)
    BLAN_errors.to_crs(32617,inplace=True)
    BLAN_errors["utmZone"] = "17N"
    BLAN_errors["itcEasting"] = BLAN_errors.geometry.apply(lambda x: x.coords[0][0])
    BLAN_errors["itcNorthing"] = BLAN_errors.geometry.apply(lambda x: x.coords[0][1])
    
    #reupdate
    shp.loc[BLAN_errors.index] = BLAN_errors
    
    #Oak Right Lab has no AOP data
    shp = shp[~(shp.siteID=="ORNL")]
    
    #resample to N examples
    shp = shp[["siteID","plotID","height","elevation","domainID","individualID","taxonID","itcEasting","itcNorthing","plantStatus","geometry"]]
    shp = shp.reset_index(drop=True)
    return shp
        
def filter_CHM(shp, lookup_glob):
        """For each plotID extract the heights from LiDAR derived CHM
        Args:
            shp: shapefile of data to filter
            lookup_glob: recursive glob search for CHM files
        """    
        filtered_results = []
        lookup_pool = glob.glob(lookup_glob, recursive=True)        
        for name, group in shp.groupby("plotID"):
            result = postprocess_CHM(group, lookup_pool=lookup_pool)
            filtered_results.append(result)
        filtered_shp = gpd.GeoDataFrame(pd.concat(filtered_results,ignore_index=True))
        
        return filtered_shp

def sample_if(x,n):
    """Sample up to n rows if rows is less than n
    Args:
        x: pandas object
        n: row minimum
        species_counts: number of each species in total data
    """
    if x.shape[0] < n:
        return x.sample(n=n, replace=True)
    else:
        return x
    
def train_test_split(ROOT, lookup_glob, n=None):
    """Create the train test split
    Args:
        ROOT: 
        lookup_glob: The recursive glob path for the canopy height models to create a pool of .tif to search
        min_diff: minimum height diff between field and CHM data
        n: number of resampled points per class
        """
    test = test_split("{}/data/raw/test_with_uid.csv".format(ROOT), field_data_path="{}/data/raw/latest_full_veg_structure.csv".format(ROOT))
    #Interpolate CHM height
    test = filter_CHM(test, lookup_glob)
    train = train_split("{}/data/raw/latest_full_veg_structure.csv".format(ROOT), test.individualID, test.taxonID.unique())
    
    filtered_train = filter_CHM(train, lookup_glob)
    filtered_train = filtered_train[filtered_train.CHM_height>1]   
    filtered_train = filtered_train[filtered_train.taxonID.isin(test.taxonID.unique())]
    test = test[test.taxonID.isin(filtered_train.taxonID.unique())]

    if not n is None:
        filtered_train  =  filtered_train.groupby("taxonID").apply(lambda x: sample_if(x,n)).reset_index(drop=True)
        
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
    #remove CHM points under 1m    
    
    check_empty = test[test.individualID.isin(train.individualID.unique())]
    assert check_empty.empty
    
    test.to_file("{}/data/processed/test.shp".format(ROOT))
    train.to_file("{}/data/processed/train.shp".format(ROOT))    
    filtered_train.to_file("{}/data/processed/CHM_filtered_train.shp".format(ROOT))    
    
    #Create files for indexing
    #Create and save a new species and site label dict
    unique_species_labels = np.concatenate([filtered_train.taxonID.unique(), test.taxonID.unique()])
    unique_species_labels = np.unique(unique_species_labels)
    
    species_label_dict = {}
    for index, label in enumerate(unique_species_labels):
        species_label_dict[label] = index
    pd.DataFrame(species_label_dict.items(), columns=["taxonID","label"]).to_csv("{}/data/processed/species_class_labels.csv".format(ROOT))    
    
    unique_site_labels = np.concatenate([filtered_train.siteID.unique(), test.siteID.unique()])
    unique_site_labels = np.unique(unique_site_labels)
    site_label_dict = {}
    for index, label in enumerate(unique_site_labels):
        site_label_dict[label] = index
    pd.DataFrame(site_label_dict.items(), columns=["siteID","label"]).to_csv("{}/data/processed/site_class_labels.csv".format(ROOT))  
    