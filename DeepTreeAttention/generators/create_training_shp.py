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
        
    return df

        
def filter_CHM(shp, lookup_glob):
        """For each plotID extract the heights from LiDAR derived CHM
        Args:
            shp: shapefile of data to filter
            lookup_glob: recursive glob search for CHM files
        """    
        filtered_results = []
        lookup_pool = glob.glob(lookup_glob, recursive=True)        
        for name, group in shp.groupby("plotID"):
            try:
                result = postprocess_CHM(group, lookup_pool=lookup_pool)
                filtered_results.append(result)
            except Exception as e:
                print("plotID {} raised: {}".format(name,e))
                
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

def sample_plots(shp):
    #split by plot level
    test_plots = shp.plotID.drop_duplicates().sample(frac=0.10)
    
    test = shp[shp.plotID.isin(test_plots)]
    train = shp[~shp.plotID.isin(test_plots)]
    
    test = test.groupby("taxonID").filter(lambda x: x.shape[0] > 5)
    
    train = train[train.taxonID.isin(test.taxonID)]
    test = test[test.taxonID.isin(train.taxonID)]

    #remove any test species that don't have site distributions in train
    to_remove = []
    for index,row in test.iterrows():
        if train[(train.taxonID==row["taxonID"]) & (train.siteID==row["siteID"])].empty:
            to_remove.append(index)
        
    add_to_train = test[test.index.isin(to_remove)]
    train = pd.concat([train, add_to_train])
    test = test[~test.index.isin(to_remove)]    
    
    return train, test

def train_test_split(ROOT=".", lookup_glob=None, n=None, debug=False):
    """Create the train test split
    Args:
        ROOT: 
        lookup_glob: The recursive glob path for the canopy height models to create a pool of .tif to search
        min_diff: minimum height diff between field and CHM data
        n: number of resampled points per class
        """
    field = pd.read_csv("{}/data/raw/2020_vst_december.csv".format(ROOT))
    field = field[~field.elevation.isnull()]
    field = field[~field.growthForm.isin(["liana","small shrub"])]
    field = field[~field.growthForm.isnull()]
    field = field[~field.plantStatus.isnull()]        
    field = field[field.plantStatus.str.contains("Live")]    
    field = field[~(field.canopyPosition.isin(["Full shade", "Mostly shaded"]))]
    field = field[(field.height > 3) | (field.height.isnull())]
    field = field[field.stemDiameter > 10]
    field = field[~field.taxonID.isin(["BETUL", "FRAXI", "HALES", "PICEA", "PINUS", "QUERC", "ULMUS", "2PLANT"])]
    field = field[~(field.eventID.str.contains("2014"))]
    field = field.groupby("individualID").apply(lambda x: x.sort_values(["eventID"],ascending=False).head(1)).reset_index(drop=True)
    
    #Create shapefile
    field["geometry"] = [Point(x,y) for x,y in zip(field["itcEasting"], field["itcNorthing"])]
    shp = gpd.GeoDataFrame(field)
    
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
    shp = shp[~(shp.siteID.isin(["PUUM","ORNL"]))]
    
    #Interpolate CHM height
    if lookup_glob:
        shp = filter_CHM(shp, lookup_glob)
        shp = shp[shp.CHM_height > 1]
        
        #remove CHM points under 4m diff  
        shp = shp[(shp.height.isnull()) | (abs(shp.height - shp.CHM_height) < 4)]  
        
    #atleast 10 data samples overall
    shp = shp.groupby("taxonID").filter(lambda x: x.shape[0] > 10)
    
    #set seed.
    np.random.seed(1)
    
    most_species = 0
    if debug:
        iterations = 1
    else:
        iterations = 300
        
    for x in np.arange(iterations):
        train, test = sample_plots(shp)
        if len(train.taxonID.unique()) > most_species:
            print(len(train.taxonID.unique()))
            saved_train = train
            saved_test = test
            most_species = len(train.taxonID.unique())
    
    train = saved_train
    test = saved_test
    
    print("There are {} records for {} species for {} sites in filtered train".format(
        train.shape[0],
        len(train.taxonID.unique()),
        len(train.siteID.unique())
    ))
    
    print("There are {} records for {} species for {} sites in test".format(
        test.shape[0],
        len(test.taxonID.unique()),
        len(test.siteID.unique())
    ))
    
    #Give tests a unique index to match against
    test["id"] = test.index.values
    train["id"] = train.index.values
    
    if not debug:    
        test.to_file("{}/data/processed/test.shp".format(ROOT))
        train.to_file("{}/data/processed/train.shp".format(ROOT))    
    
        #Create files for indexing
        #Create and save a new species and site label dict
        unique_species_labels = np.concatenate([train.taxonID.unique(), test.taxonID.unique()])
        unique_species_labels = np.unique(unique_species_labels)
        
        species_label_dict = {}
        for index, label in enumerate(unique_species_labels):
            species_label_dict[label] = index
        pd.DataFrame(species_label_dict.items(), columns=["taxonID","label"]).to_csv("{}/data/processed/species_class_labels.csv".format(ROOT))    
        
        unique_site_labels = np.concatenate([train.siteID.unique(), test.siteID.unique()])
        unique_site_labels = np.unique(unique_site_labels)
        site_label_dict = {}
        for index, label in enumerate(unique_site_labels):
            site_label_dict[label] = index
        pd.DataFrame(site_label_dict.items(), columns=["siteID","label"]).to_csv("{}/data/processed/site_class_labels.csv".format(ROOT))  
        
        unique_domain_labels = np.concatenate([train.domainID.unique(), test.domainID.unique()])
        unique_domain_labels = np.unique(unique_domain_labels)
        domain_label_dict = {}
        for index, label in enumerate(unique_domain_labels):
            domain_label_dict[label] = index
        pd.DataFrame(domain_label_dict.items(), columns=["domainID","label"]).to_csv("{}/data/processed/domain_class_labels.csv".format(ROOT))  
        
        