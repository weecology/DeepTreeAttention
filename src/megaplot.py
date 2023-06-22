#Megaplot data processing
import geopandas as gpd
import glob
import os
import numpy as np
import pandas as pd
from src import CHM
import shapely
from distributed import wait

def read_files(directory, site=None, config=None, client=None):
    """Read shapefiles and return a dict based on site name"""
    shapefiles = glob.glob("{}/*.shp".format(directory))
    if site:
        shapefiles = [x for x in shapefiles if site in x]            
    shps = [gpd.read_file(x) for x in shapefiles]
    sites = [os.path.splitext(os.path.basename(x))[0] for x in shapefiles]
    
    sitedf = []
    
    if client:
        futures = []
        for index, x in enumerate(sites):
            future = client.submit(format, site=x, gdf=shps[index], config=config)
            futures.append(future)
        wait(futures)
        for x in futures:
            print(x)            
            formatted_data = x.result()
            formatted_data.set_crs("32616", inplace=True, allow_override=True)
            sitedf.append(formatted_data)        
    else: 
        for index, x in enumerate(sites):
            print(x)
            formatted_data = format(site=x, gdf=shps[index], config=config)
            # Set as dummy crs, since different utm zones
            formatted_data.set_crs("32616", inplace=True, allow_override=True)
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
    gdf["individual"] = gdf.index.to_series().apply(lambda x: "{}.contrib.{}".format(site,x)) 
    gdf["filename"] = site
    gdf["siteID"] = site.split("_")[0]
    
    #PlotID variable to center on correct tile
    if gdf.shape[0] > 1000:
        grid = create_grid(gdf)
        gdf = gpd.sjoin(gdf, grid)
    else:
        gdf = buffer_plots(gdf)
    
    #Make sure any points sitting on the line are assigned only to one grid. Rare edge case
    gdf = gdf.groupby("individual").apply(lambda x: x.head(1)).reset_index(drop=True)
    
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
    grid["plotID"] = grid.plotID.apply(lambda x: "{}_contrib_{}".format(gdf.filename.unique()[0], int(x)))
    
    return grid
    
def load(directory, config, client=None, site=None):
    """Load all the megaplot data and generate crown predictions
    Args:
        directory: location of .csv files of megaplot data
        client: optional dask client
        site: a list of sites to include, "all" or None will bypass filter.
    Returns:
        crowndf: a geopandas dataframe of crowns for all sites
    """
    if site == "pretrain":
        site = None
        
    if site is not None:
        if type(site) is not list:
            raise TypeError("site parameter should be a list of strings")
        all_sites = []
        for x in site:
            df = read_files(directory=directory, config=config, client=client, site=x)
            all_sites.append(df)
        formatted_data = pd.concat(all_sites)
    else:
        formatted_data = read_files(directory=directory, config=config, client=client, site=site)
    
    return formatted_data