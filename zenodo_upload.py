### Zenodo upload, need to source .zenodo_token, module load jq
import requests
import glob
import os
import shutil
from src.model_list import species_model_paths
import subprocess

def upload(path):
    """Upload an item to zenodo"""
    call = "/home/b.weinstein/zenodo-upload/zenodo_upload.sh 10035099 {}".format(path)
    print(call)
    subprocess.call(call,shell=True)

if __name__== "__main__":
    files_to_upload = []
    #files_to_upload.append("requirements.txt")
    crop_zip = "/blue/ewhite/b.weinstein/DeepTreeAttention/fba8ff88ef834016a335e8ce07f38131.zip"
    files_to_upload.append(crop_zip)
    file_sizes = 0
    for site in species_model_paths:
        if site == "STEI":
            continue
        model_path = species_model_paths[site]
        basename = os.path.splitext(os.path.basename(model_path))[0]
        zip_path = "/blue/ewhite/b.weinstein/DeepTreeAttention/results/predictions/{}/{}/{}.zip".format(site, basename, site)
        files_to_upload.append(zip_path)
        csv_path = "/blue/ewhite/b.weinstein/DeepTreeAttention/results/predictions/{}/{}/{}.csv".format(site, basename, site)
        files_to_upload.append(csv_path)
    for f in files_to_upload:     
        print(f)   
        #upload(f)
        basename = os.path.basename(f)
        dst_path = os.path.join("/blue/ewhite/b.weinstein/DeepTreeAttention/Zenodo/", basename) 
        shutil.copy(f, dst_path)
        file_sizes += os.path.getsize(f)
    print("Total file size is {}".format(file_sizes))
