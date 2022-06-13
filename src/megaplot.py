#Megaplot data processing
import geopandas as gpd
import glob
import os
import numpy as np
import pandas as pd
from src import CHM
import shapely

def read_files(directory, site=None, config=None):
    """Read shapefiles and return a dict based on site name"""
    shapefiles = glob.glob("{}/*.shp".format(directory))
    if site:
        shps = [x for x in shapefiles if site in x]            
    shps = [gpd.read_file(x) for x in shapefiles]
    sites = [os.path.splitext(os.path.basename(x))[0] for x in shapefiles]
    
    sitedf = []
    for index, x in enumerate(sites):
        print(x)
        formatted_data = format(site=x, gdf=shps[index], config=config)
        sitedf.append(formatted_data)

    sitedf = pd.concat(sitedf)
    
    return sitedf

def format(site, gdf, config):
    """The goal of this function is to mimic for the format needed to input to generate.points_to_crowns. 
    This requires a plot ID, individual, taxonID and site column. The individual should encode siteID and year
    Args:
        site: siteID
        gdf: site data
    """
    #give each an individual ID
    gdf["individualID"] = gdf.index.to_series().apply(lambda x: "{}.contrib.{}".format(site,x)) 
    gdf["filename"] = site
    gdf["siteID"] = site.split("_")[0]
    
    #PlotID variable to center on correct tile
    if gdf.shape[0] > 1000:
        grid = create_grid(gdf)
        gdf = gpd.sjoin(gdf, grid)
    else:
        gdf = buffer_plots(gdf)
    
    #Make sure any points sitting on the line are assigned only to one grid. Rare edge case
    gdf = gdf.groupby("individualID").apply(lambda x: x.head(1)).reset_index(drop=True)
    
    if "height" in gdf.columns: 
        #Height filter 
        gdf = CHM.filter_CHM(gdf, CHM_pool=config["CHM_pool"],max_CHM_diff=config["max_CHM_diff"], min_CHM_height=config["min_CHM_height"], CHM_height_limit=config["CHM_height_limit"])      
        
    return gdf

def buffer_plots(gdf):
    plotID = 0
    for x in gdf.geometry.centroid:
        x_buffer = x.buffer(40)
        touches = gdf.geometry.centroid.intersection(x_buffer)
        touches = touches[~touches.is_empty]
        if not touches.empty:    
            gdf.loc[touches.index, "plotID"] = plotID
            plotID +=1
    gdf["plotID"] = gdf.plotID.apply(lambda x: "{}_contrib_{}".format(gdf.filename.unique()[0], int(x)))
    
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
    for x0 in np.arange(xmin, xmax+cell_size, cell_size):
        for y0 in np.arange(ymin, ymax+cell_size, cell_size):
            # bounds
            x1 = x0-cell_size
            y1 = y0+cell_size
            grid_cells.append(shapely.geometry.box(x0, y0, x1, y1)  )
    grid = gpd.GeoDataFrame(grid_cells, columns=['geometry'],crs=crs)
    
    #give grid cells a plot ID
    grid["plotID"] = grid.index
    grid["plotID"] = grid.plotID.apply(lambda x: "{}_contrib".format(x))
    
    return grid
    
def load(directory, config, site=None):
    """Load all the megaplot data and generate crown predictions
    Args:
        directory: location of .csv files of megaplot data
        client: optional dask client
    Returns:
        crowndf: a geopandas dataframe of crowns for all sites
    """
    formatted_data = read_files(directory=directory, config=config, site=site)
    
    return formatted_data