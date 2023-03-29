#Utility functions for searching for NEON schema data given a bound or filename. Optionally generating .tif files from .h5 hyperspec files.
import os
import math
import re
import h5py
import numpy as np
from src import Hyperspectral
import rasterio as rio

def bounds_to_geoindex(bounds):
    """Convert an extent into NEONs naming schema
    Args:
        bounds: list of top, left, bottom, right bounds, usually from geopandas.total_bounds
    Return:
        geoindex: str {easting}_{northing}
    """
    easting = int(np.mean([bounds[0], bounds[2]]))
    northing = int(np.mean([bounds[1], bounds[3]]))

    easting = math.floor(easting / 1000) * 1000
    northing = math.floor(northing / 1000) * 1000

    geoindex = "{}_{}".format(easting, northing)

    return geoindex

def find_sensor_path(lookup_pool, shapefile=None, bounds=None, geo_index=None, all_years=False):
    """Find a hyperspec path based on the shapefile using NEONs schema
    Args:
        bounds: Optional: list of top, left, bottom, right bounds, usually from geopandas.total_bounds. Instead of providing a shapefile
        lookup_pool: glob string to search for matching files for geoindex
    Returns:
        match: full path to sensor tile, if all years, a list of paths
    """
    if not geo_index:
        if shapefile:
            basename = os.path.splitext(os.path.basename(shapefile))[0]
            geo_index = re.search("(\d+_\d+)_image", basename).group(1)
        else:
            geo_index = bounds_to_geoindex(bounds=bounds) 
    
    match = [x for x in lookup_pool if geo_index in x]
    
    if len(match) == 0:
        raise ValueError("No matches for geoindex {} in sensor pool".format(geo_index))                    
        
    #Get most recent year
    if all_years:
        return match
    else:        
        match.sort()
        match = match[::-1]
        
        return match[0]


def convert_h5(hyperspectral_h5_path, rgb_path, savedir, year=None):
    if year:
        tif_basename = os.path.splitext(os.path.basename(rgb_path))[0] + "_hyperspectral_{}.tif".format(year)
    else:
        tif_basename = os.path.splitext(os.path.basename(rgb_path))[0] + "_hyperspectral.tif"
    tif_path = "{}/{}".format(savedir, tif_basename)

    Hyperspectral.generate_raster(h5_path=hyperspectral_h5_path,
                                  rgb_filename=rgb_path,
                                  suffix=year,
                                  bands="no_water",
                                  save_dir=savedir)

    return tif_path


def lookup_and_convert(rgb_pool, h5_pool, savedir, bounds = None, geo_index=None, shapefile=None, all_years=False):
    """For a given geo_index bounds, lookup the h5 tile and convert to .tif"""
    hyperspectral_h5_path = find_sensor_path(shapefile=shapefile,lookup_pool=h5_pool, bounds=bounds, geo_index=geo_index, all_years=all_years)
    rgb_path = find_sensor_path(shapefile=shapefile, lookup_pool=rgb_pool, bounds=bounds, geo_index=geo_index)

    if type(hyperspectral_h5_path) == list:
        tif_paths = []
        for x in hyperspectral_h5_path:
            #convert .h5 hyperspec tile if needed
            year = year_from_tile(x)
            tif_basename = os.path.splitext(os.path.basename(rgb_path))[0] + "_hyperspectral_{}.tif".format(year)
            tif_path = "{}/{}".format(savedir, tif_basename)
            # try to open file to confirm it exists and is valid.
            try:
                r = rio.open(tif_path)
            except:
                tif_path = convert_h5(x, rgb_path, savedir, year=year)                  
            tif_paths.append(tif_path)
                
        return tif_paths
    else:
        #convert .h5 hyperspec tile if needed
        tif_basename = os.path.splitext(os.path.basename(rgb_path))[0] + "_hyperspectral.tif"
        tif_path = "{}/{}".format(savedir, tif_basename)
    
        if not os.path.exists(tif_path):
            tif_path = convert_h5(hyperspectral_h5_path, rgb_path, savedir)
    
        return tif_path

def year_from_tile(path):
    return path.split("/")[6]

def site_from_path(path):
    basename = os.path.splitext(os.path.basename(path))[0]
    site_name = re.search("NEON_D\d+_(\w+)_D", basename).group(1)
    
    return site_name

def domain_from_path(path):
    basename = os.path.splitext(os.path.basename(path))[0]
    domain_name = re.search("NEON_(D\d+)_\w+_D", basename).group(1)
    
    return domain_name

def elevation_from_tile(path):
    try:
        h5 = h5py.File(path, 'r')
        elevation = h5[list(h5.keys())[0]]["Reflectance"]["Metadata"]["Ancillary_Imagery"]["Smooth_Surface_Elevation"].value.mean()
        h5.close()
    except Exception as e:
        raise IOError("{} failed to read elevation from tile:".format(path, e))
 
    return elevation

    