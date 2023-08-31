import glob
import os
from src.model_list import species_model_paths

#files = glob.glob("/blue/ewhite/b.weinstein/DeepTreeAttention/results/site_crops/*/shp/*.shp")

#for f in files:
#    basename = os.path.splitext(os.path.basename(f))[0]
#    tar_name = f.replace("/shp","/tar").replace(".shp",".tar.gz")
#    if not os.path.exists(tar_name):
#        print(f)
#        os.remove(f)


for site in species_model_paths:
    model_name = os.path.splitext(os.path.basename(species_model_paths[site]))[0]
    files = glob.glob("/blue/ewhite/b.weinstein/DeepTreeAttention/results/predictions/{}/{}/*.shp".format(site, model_name))
    for f in files:
        basename = os.path.splitext(os.path.basename(f))[0]
        shp_name = "/blue/ewhite/b.weinstein/DeepTreeAttention/results/site_crops/{}/shp/{}.shp".format(site, basename)
    if not os.path.exists(shp_name):
        print(f)
        #os.remove(f)