# Download models and notebooks
import subprocess
import shutil
import os
import glob
import zipfile
import geopandas as gpd
from src import start_cluster
from distributed import wait
import pandas as pd

# cleanup shapefiles

def clean_up(path):
    a = gpd.read_file(path)
    try:
        a = a.drop(columns=["conifer_la", "broadleaf_",
                   "dominant_c", "oak_label", "ensembleTa", "ens_label"])
        a = a.rename(columns={"broadlea_1": "bleaf_score",
                              "conifer_sc": "conif_score",
                              "dominant_1": "dom_score",
                              "dominant_2": "dom_taxa",
                              "conifer_ta": "conif_taxa",
                              "broadlea_2": "bleaf_taxa"})
    except:
        pass

    a.to_file(path)

    return a


client = start_cluster.start(cpus=30)
species_model_paths = {
    "NIWO": "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/def58fe6c0fa4b8991e5e80f63a20acd_NIWO.pt",
    "RMNP": "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/bd761ac1c0d74268a59e87aa85b9fa9c_RMNP.pt",
    "SJER": "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/702f6a7cf1b24307b8a23e25148f7559_SJER.pt",
    "WREF": "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/a5453bab67b14eb8b01da57f79590409_WREF.pt",
    "SERC": "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/95f74a748e2f48d5bed0635228fed41a_SERC.pt",
    "GRSM": "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/127191a057e64a7fb7267cc889e56c25_GRSM.pt",
    "DEJU": "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/0e3178ac37434aeb90ac207c18a9caf7_DEJU.pt",
    "BONA": "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/2975c34a7ca540df9c54d261ef95551e_BONA.pt",
    "TREE": "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/f191ef1bb2d54d459eb4f12d73756822_TREE.pt",
    "STEI": "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/951f766a48aa4a0baa943d9f6d26f3e0_STEI.pt",
    "UNDE": "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/185caf9f910c4fd3a7f5e470b6828090_UNDE.pt",
    "DELA": "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/1d22265240bc4e52b77df78ce557e40f_DELA.pt",
    "LENO": "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/537b80fd46e64c77a1c367dcbef713e3_LENO.pt",
    "OSBS": "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/9fca032e2322479b82506e700de065f5_OSBS.pt",
    "JERC": "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/86d51ae4b7a34308bc99c19f8eeadf41_JERC.pt",
    "TALL": "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/c52006e0461b4363956335203e42f786_TALL.pt",
    "CLBJ": "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/83c1d8fd4c69479185ed3224abb6e8f9_CLBJ.pt",
    "TEAK": "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/b9b6882705d24fe6abf12282936ababb_TEAK.pt",
    "SOAP": "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/97695ef8ec6a481fb3515d29d2cf33bb_SOAP.pt",
    "YELL": "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/e1760a6a288747beb83f055155e49109_YELL.pt",
    "MLBS": "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/59f4175b26fc48f8b6f16d0598d49411_MLBS.pt",
    "BLAN": "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/ce50bb593a484a28b346e7efe357e0fa_BLAN.pt",
    "UKFS": "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/1d9305235bf44f319def7915a4b7b21f_UKFS.pt",
    "BART": "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/01bfce34aacd455bb1a4b4d80deb16d2_BART.pt",
    "HARV": "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/973a01b0c1a349ebaf4fc8454ffc624d_HARV.pt"}

# Clean up the files for each site
for site in ["BART"]:
    basename = os.path.splitext(os.path.basename(species_model_paths[site]))[0]
    predictions = glob.glob(
        "/blue/ewhite/b.weinstein/DeepTreeAttention/results/predictions/{}/{}/*.shp".format(site, basename), recursive=True)
    futures = client.map(clean_up, predictions)
    wait(futures)

    site_csv = []
    for future in futures:
        result = future.result()
        site_csv.append(pd.DataFrame(result))
    site_csv = pd.concat(site_csv)
    site_csv.to_csv(
        "/blue/ewhite/b.weinstein/DeepTreeAttention/results/predictions/{}/{}/{}.csv".format(site, basename, site))

zipfilename = "/blue/ewhite/b.weinstein/DeepTreeAttention/results/predictions/{}/{}/{}.zip".format(
    site, basename, site)
# remove old
try:
    os.remove(zipfilename)
except:
    pass

with zipfile.ZipFile(zipfilename, "w") as zipObj:
    for folderName, subfolders, filenames in os.walk("/blue/ewhite/b.weinstein/DeepTreeAttention/results/predictions/{}/{}".format(site, basename)):
        for filename in filenames:
            # create complete filepath of file in directory
            filePath = os.path.join(folderName, filename)

            # Don't include itself
            if ".zip" in filePath:
                continue
            # Add file to zip
            print(filePath)
            zipObj.write(filePath, arcname=filename)