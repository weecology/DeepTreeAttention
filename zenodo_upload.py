### Zenodo upload, need to source .zenodo_token, module load jq
import requests
import glob
import os
from src.model_list import species_model_paths
import subprocess

def upload(path):
    """Upload an item to zenodo"""
    call = "/home/b.weinstein/zenodo-upload/zenodo_upload.sh 8253261 {}".format(path)
    subprocess.call(call,shell=True)

if __name__== "__main__":
    files_to_upload = []
    #files_to_upload.append("requirements.txt")
    crop_zip = "/blue/ewhite/b.weinstein/DeepTreeAttention/fba8ff88ef834016a335e8ce07f38131.zip"
    files_to_upload.append(crop_zip)
    for site in species_model_paths:
        model_path = species_model_paths[site]
        basename = os.path.basename(model_path)
        zip_path = "/blue/ewhite/b.weinstein/DeepTreeAttention/results/predictions/{}/{}/{}.zip".format(site, basename, site)
        files_to_upload.append(zip_path)
        csv_path = "/blue/ewhite/b.weinstein/DeepTreeAttention/results/predictions/{}/{}/{}.csv".format(site, basename, site)
        files_to_upload.append(csv_path)

    for f in files_to_upload:     
        print(f)   
        upload(f)