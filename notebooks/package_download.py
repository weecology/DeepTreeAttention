# Download models and notebooks
import subprocess
import shutil
import os
import glob
import zipfile
import geopandas as gpd
from src import start_cluster
from src.model_list import species_model_paths
from distributed import wait, Client
import pandas as pd

# cleanup shapefiles
def clean_up(path):
    a = gpd.read_file(path)
    try:
        a = a.drop(columns=["conifer_la", "broadleaf_",
                   "dominant_c", "oak_label", "ensembleTa", "ens_label"], errors="ignore")
        a = a.rename(columns={"broadlea_1": "bleaf_score",
                              "conifer_sc": "conif_score",
                              "dominant_1": "dom_score",
                              "dominant_2": "dom_taxa",
                              "conifer_ta": "conif_taxa",
                              "broadlea_2": "bleaf_taxa",
                              "scientific":"sci_name",
                              "siteID":"site_id",
                              "individual":"indiv_id"})
    except:
        pass
    a.to_file(path)
    # Reproject for earth engine
    b = a.to_crs("EPSG:4326")
    return b

#client = start_cluster.start(cpus=50,mem_size="5GB")
#client = Client()

# Clean up the files for each site
for site in species_model_paths: 
    print(site)
    zipfilename = "/blue/ewhite/b.weinstein/DeepTreeAttention/Zenodo/{}.zip".format(
        site)
    if os.path.exists(zipfilename):
        continue
    basename = os.path.splitext(os.path.basename(species_model_paths[site]))[0]
    predictions = glob.glob(
        "/blue/ewhite/b.weinstein/DeepTreeAttention/results/predictions/{}/{}/draped/*.shp".format(site, basename), recursive=True)
    if len(predictions) == 0:
        continue
    #futures = client.map(clean_up, predictions)
    #wait(futures)
    #site_csv = []
    #for future in futures:
    #    result = future.result()
    #    site_csv.append(pd.DataFrame(result))
    site_shps = []
    for x in predictions:
        gdf = gpd.read_file(x)
        site_shps.append(gdf)
        
    site_csv = pd.concat(site_shps)
    site_csv.to_csv(
        "/blue/ewhite/b.weinstein/DeepTreeAttention/results/predictions/{}/{}/draped/{}.csv".format(site, basename, site), index=False)
    zipfilename = "/blue/ewhite/b.weinstein/DeepTreeAttention/Zenodo/{}.zip".format(
        site)
    
    file_locations = "/blue/ewhite/b.weinstein/DeepTreeAttention/results/predictions/{}/{}/draped/*".format(site, basename)
    print(file_locations)
    subprocess.call("zip -9 -j -r {} {}".format(zipfilename, file_locations), shell=True)
