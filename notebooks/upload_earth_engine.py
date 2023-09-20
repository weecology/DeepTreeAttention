# Earth engine upload, seeded by chatgpt
import os
import json
from google.cloud import storage
import subprocess
from src.model_list import species_model_paths

"""
This module assumes that predictions have already been uploaded using gsutil to a google cloud bucket.
gsutil and earthengine authenticate
"""   

#for site, model in species_model_paths.items():
#    model_path = os.path.basename(os.path.splitext(model)[0])
#    csv_path = "/blue/ewhite/b.weinstein/DeepTreeAttention/results/predictions/{}/{}/{}.csv".format(site, model_path, site)
#    subprocess.call("gsutil cp {} gs://earthengine_shapefiles/".format(csv_path), shell=True)

def create_csv_manifest(site):
    client = storage.Client()
    csvs = []
    for blob in client.list_blobs('earthengine_shapefiles', prefix=site):
        if str(blob.name).endswith("csv"):
            csvs.append(blob.name)
    print(csvs)
    if len(csvs) == 0:
        return None
    # Iterate over files in the directory
    for csv in csvs:    
        # Save shapefile list as JSON
        basename = os.path.splitext(os.path.basename(csv))[0]
        subprocess.call("earthengine upload table --asset_id=users/benweinstein2010/{} gs://earthengine_shapefiles/{}".format(basename, csv), shell=True)

for x in ["BART","HARV"]:
    create_csv_manifest(site=x)