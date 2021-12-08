#spatial neighbors

def spatial_neighbors(gdf, n):
    """    
    #Get all neighbors within n meters of each point.
    Args:
        gdf: a geodataframe
        buffer: distance from focal point in m to search for neighbors
    Returns:
        neighbors: dictionary with keys -> index of the gdf, value of index of neighbors
    """
    neighbors = {}
    for x in gdf.index:
        geom = gdf.loc[x].geometry.centroud.buffer(buffer=n)
        touches = gdf.intersection(geom)
        touches = touches[~(touches.geometry.is_empty)]
        #remove itself
        neighbors[x] = [i for i in touches.index if not x == i]
        
    return neighbors