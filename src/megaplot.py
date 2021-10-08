#Megaplot data processing
import geopandas as gpd
import glob
import os
import numpy as np
import pandas as pd
from src import generate
import shapely

def read_files(directory):
    """Read shapefiles and return a dict based on site name"""
    shapefiles = glob.glob("{}/*.shp".format(directory))
    shapefiles = [x for x in shapefiles if not "points" in x]
    shps = [gpd.read_file(x) for x in shapefiles]
    sites = [os.path.splitext(os.path.basename(x))[0] for x in shapefiles]
    
    site_dict = {}
    for index, x in enumerate(sites):
        formatted_data = format(site=x, gdf=shps[index], directory=directory)
        site_dict[x] = formatted_data

    return site_dict

def format(site, gdf, directory):
    """The goal of this function is to mimic for the format needed to input to generate.points_to_crowns. 
    This requires a plot ID, individual, taxonID and site column. The individual should encode siteID and year
    Args:
        site: siteID
        gdf: site data
    """
    species_data = pd.read_csv("{}/{}.csv".format(directory, site))
    species_data = species_data.dropna(subset=["taxonID"])
    gdf = gdf.merge(species_data[["sp","taxonID"]])
    
    #give each an individual ID
    gdf["individualID"] = gdf.apply(lambda x: "{}_{}".format(site,x.index),axis=1) 
    gdf["siteID"] = site
    
    #PlotID variable to center on correct tile
    grid = create_grid(gdf)
    gdf = gpd.sjoin(gdf, grid)
       
    return gdf

def create_grid(gdf):
    """Create a rectangular grid that overlays a geopandas object"""
    xmin, ymin, xmax, ymax= gdf.total_bounds
    # how many cells across and down
    cell_size = 40
    # projection of the grid
    crs = gdf.crs
    # create the cells in a loop
    grid_cells = []
    for x0 in np.arange(xmin, xmax+cell_size, cell_size ):
        for y0 in np.arange(ymin, ymax+cell_size, cell_size):
            # bounds
            x1 = x0-cell_size
            y1 = y0+cell_size
            grid_cells.append( shapely.geometry.box(x0, y0, x1, y1)  )
    grid = gpd.GeoDataFrame(grid_cells, columns=['geometry'],crs=crs)
    #give grid cells a plot ID
    grid["plotID"] = grid.index
    
    return grid
    
def load(directory, rgb_pool,client):
    """Load all the megaplot data and generate crown predictions
    Args:
        directory: location of .csv files of megaplot data
        rgb_pool: glob path location to search for rgb files
        client: optional dask client
    Returns:
        crowndf: a geopandas dataframe of crowns for all sites
    """
    formatted_data = read_files(directory=directory)
    crown_list = []
    for x in formatted_data:
        formatted_data[x].to_file("{}/{}_points.shp".format(directory, x))
        crowns = generate.points_to_crowns(
            field_data="{}/{}_points.shp".format(directory, x),
            rgb_dir=rgb_pool,
            savedir=None,
            client=client,
            raw_box_savedir=None)
        crown_list.append(crowns)
    
    crowndf = gpd.GeoDataFrame(pd.concat(crown_list))
    
    return crowndf