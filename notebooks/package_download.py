# Download models and notebooks
import subprocess
import shutil
import os
import glob
import zipfile
import geopandas as gpd
from src import start_cluster
from src.model_list import species_model_paths
from distributed import wait
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

client = start_cluster.start(cpus=40)

# Clean up the files for each site
for site in ["TEAK"]:
    print(site)
    basename = os.path.splitext(os.path.basename(species_model_paths[site]))[0]
    predictions = glob.glob(
        "/blue/ewhite/b.weinstein/DeepTreeAttention/results/predictions/{}/{}/*.shp".format(site, basename), recursive=True)
    if len(predictions) == 0:
        continue
    futures = client.map(clean_up, predictions)
    wait(futures)
    site_csv = []
    for future in futures:
        result = future.result()
        site_csv.append(pd.DataFrame(result))
    site_csv = pd.concat(site_csv)
    site_csv.to_csv(
        "/blue/ewhite/b.weinstein/DeepTreeAttention/results/predictions/{}/{}/{}.csv".format(site, basename, site), index=False)
    zipfilename = "/blue/ewhite/b.weinstein/DeepTreeAttention/results/predictions/{}/{}/{}.zip".format(
        site, basename, site)
    with zipfile.ZipFile(zipfilename, "w") as zipObj:
        print("creating zip")
        for folderName, subfolders, filenames in os.walk("/blue/ewhite/b.weinstein/DeepTreeAttention/results/predictions/{}/{}".format(site, basename)):
            for filename in filenames:
                # create complete filepath of file in directory
                filePath = os.path.join(folderName, filename)
                # Don't include itself
                if ".zip" in filePath:
                    continue
                # Add file to zip
                zipObj.write(filePath, arcname=filename)