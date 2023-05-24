from src import start_cluster
import geopandas
import pandas
from dask import delayed
from dask import distributed as dd
import glob


client = start_cluster.start(cpus=50)

def read_shp(path):
    gdf = geopandas.read_file(path)
    df = pd.DataFrame(gdf)
    
    return df

shps = glob.glob("/blue/ewhite/b.weinstein/DeepTreeAttention/results/predictions/**/*.shp", recursive=True)
dfs = [read_shp(x) for x in shps]
ddf = dd.from_delayed(dfs)

#How many species
ddf.scientific.unique().compute()

#How many trees
ddf.shape[0].compute()
