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

client = start_cluster.start(cpus=80)


species_model_paths = {
    "NIWO": "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/000b1ecf0ca6484893e177e3b5d42c7e_NIWO.pt",
    "RMNP": "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/b6ceb35a3c9c4cc98241ba00ff12ff87_RMNP.pt",    
    "SJER":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/ecfdd5bf772a40cab89e89fa1549f13b_SJER.pt",
    "WREF":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/686204cb0d5343b0b20613a6cf25f69b_WREF.pt",
    "SERC":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/e5055fe5f4b8403cbc48b16d903533e0_SERC.pt",
    "GRSM":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/970ad2293e7f4ecb969a8338f7fcd76e_GRSM.pt",
    "DEJU":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/aba32c72d6bd4747abfa0d5cfbba230d_DEJU.pt",
    "BONA":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/152a61614a4a48cf84b27f5880692230_BONA.pt",
    "TREE":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/3201f8a710a24d7b891351fddfa0bf32_TREE.pt",
    "STEI":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/35220561e5834d03b1d098c84a00a171_STEI.pt",
    "UNDE":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/a6dc9a627a7446acbc40c5b7913a45e9_UNDE.pt",
    "DELA":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/5d3578860bfd4fd79072de872452ea79_DELA.pt",
    "LENO":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/b7d77034801c49dcab5b922b5704cb9e_LENO.pt",
    "OSBS":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/00fef05fa70243f1834ee437406150f7_OSBS.pt",
    "JERC":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/86d51ae4b7a34308bc99c19f8eeadf41_JERC.pt",
    "TALL":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/be884a1ac14d4379b52d25903acc7498_TALL.pt",
    "CLBJ":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/c1e0192b0f43455aadbad593cba0b356_CLBJ.pt",
    "TEAK":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/1077f3429ed84a28aa9b63b1c950a1f2_TEAK.pt",
    "SOAP":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/c56b937a94f84e9da774370f4e46a110_SOAP.pt",
    "YELL":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/f2c069f59b164795af482333a5e7fffb_YELL.pt",                       
    "MLBS":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/b5efc0037529431092db587727fb4fe9_MLBS.pt",
    "BLAN":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/533e410797c945618c72b2a54176ed61_BLAN.pt",
    "UKFS":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/084b83c44d714f23b9d96e0a212f11f1_UKFS.pt",
    "BART":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/bb0f7415e5ba46b7ac9dbadee4a141f3_BART.pt",
    "HARV":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/9130a6b5ce544e1280283bf60cab63b0_HARV.pt"}

# Clean up the files for each site
for site in species_model_paths:
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