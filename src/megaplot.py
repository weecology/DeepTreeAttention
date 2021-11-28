#Megaplot data processing
import geopandas as gpd
import glob
import os
import numpy as np
import pandas as pd
from src import generate
from src import CHM
import shapely

def read_files(directory, config=None):
    """Read shapefiles and return a dict based on site name"""
    shapefiles = glob.glob("{}/*.shp".format(directory))
    shapefiles = [x for x in shapefiles if not "points" in x]
    shps = [gpd.read_file(x) for x in shapefiles]
    sites = [os.path.splitext(os.path.basename(x))[0] for x in shapefiles]
    
    sitedf = []
    for index, x in enumerate(sites):
        print(x)
        formatted_data = format(site=x, gdf=shps[index], directory=directory, config=config)
        sitedf.append(formatted_data)

    sitedf = pd.concat(sitedf)
    
    return sitedf

def format(site, gdf, directory, config):
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
    gdf["individualID"] = gdf.index.to_series().apply(lambda x: "{}_contrib{}".format(site,x)) 
    gdf["siteID"] = site
    
    #PlotID variable to center on correct tile
    grid = create_grid(gdf)
    gdf = gpd.sjoin(gdf, grid)
    
    if "height" in gdf.columns: 
        #Height filter 
        gdf = CHM.filter_CHM(gdf, CHM_pool=config["CHM_pool"],min_CHM_diff=config["min_CHM_diff"], min_CHM_height=config["min_CHM_height"])      
        
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
    grid["plotID"] = "{}_contrib".format(grid.index)
    
    return grid
    
def load(directory, rgb_pool,client, config):
    """Load all the megaplot data and generate crown predictions
    Args:
        directory: location of .csv files of megaplot data
        rgb_pool: glob path location to search for rgb files
        client: optional dask client
    Returns:
        crowndf: a geopandas dataframe of crowns for all sites
    """
    formatted_data = read_files(directory=directory, config=config)
    
    return formatted_data