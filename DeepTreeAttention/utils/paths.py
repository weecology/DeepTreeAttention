#Utility functions for searching for NEON schema data given a bound or filename. Optionally generating .tif files from .h5 hyperspec files.
import os
import glob
import math
import re

from DeepTreeAttention.utils import Hyperspectral


def bounds_to_geoindex(bounds):
    """Convert an extent into NEONs naming schema
    Args:
        bounds: list of top, left, bottom, right bounds, usually from geopandas.total_bounds
    Return:
        geoindex: str {easting}_{northing}
    """
    easting = min(bounds[0], bounds[2])
    northing = min(bounds[1], bounds[3])

    easting = math.floor(easting / 1000) * 1000
    northing = math.floor(northing / 1000) * 1000

    geoindex = "{}_{}".format(easting, northing)

    return geoindex


def find_sensor_path(lookup_pool, shapefile=None, bounds=None):
    """Find a hyperspec path based on the shapefile using NEONs schema
    Args:
        bounds: Optional: list of top, left, bottom, right bounds, usually from geopandas.total_bounds. Instead of providing a shapefile
        lookup_pool: glob string to search for matching files for geoindex
    Returns:
        year_match: full path to sensor tile
    """

    if shapefile is None:
        geo_index = bounds_to_geoindex(bounds=bounds)
        match = [x for x in lookup_pool if geo_index in x]
        match.sort()
        try:
            year_match = match[-1]
        except Exception as e:
            raise ValueError("No matches for geoindex {} in sensor pool".format(geo_index))
    else:

        #Get file metadata from name string
        basename = os.path.splitext(os.path.basename(shapefile))[0]
        geo_index = re.search("(\d+_\d+)_image", basename).group(1)
        match = [x for x in pool if geo_index in x]
        year = re.search("(\d+)_", basename).group(1)
        year_match = [x for x in match if year in x]
        try:
            year_match = year_match[0]
        except Exception as e:
            raise ValueError("No matches for {} in {}: {}".format(
                shapefile, lookup_pool, e))

    return year_match


def convert_h5(hyperspectral_h5_path, rgb_path, savedir):
    tif_basename = os.path.splitext(os.path.basename(rgb_path))[0] + "_hyperspectral.tif"
    tif_path = "{}/{}".format(savedir, tif_basename)

    if not os.path.exists(tif_path):
        Hyperspectral.generate_raster(h5_path=hyperspectral_h5_path,
                                      rgb_filename=rgb_path,
                                      bands="All",
                                      save_dir=savedir)

    return tif_path


def lookup_and_convert(shapefile, rgb_pool, hyperspectral_pool, savedir):
    hyperspectral_h5_path = find_sensor_path(shapefile=shapefile,
                                             lookup_pool=hyperspectral_pool,
                                             sensor="hyperspectral")
    rgb_path = find_sensor_path(shapefile=shapefile, lookup_pool=rgb_pool, sensor="rgb")

    #convert .h5 hyperspec tile if needed
    tif_basename = os.path.splitext(os.path.basename(rgb_path))[0] + "_hyperspectral.tif"
    tif_path = "{}/{}".format(savedir, tif_basename)

    if not os.path.exists(tif_path):
        tif_path = convert_h5(hyperspectral_h5_path, rgb_path, savedir)

    return tif_path
