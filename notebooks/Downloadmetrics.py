#Download hyperparameters and metrics 
from comet_ml import API
import os
import pandas as pd

species_model_paths = {
    "NIWO": "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/def58fe6c0fa4b8991e5e80f63a20acd_NIWO.pt",
    "RMNP": "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/bd761ac1c0d74268a59e87aa85b9fa9c_RMNP.pt",    
    "SJER":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/702f6a7cf1b24307b8a23e25148f7559_SJER.pt",
    "WREF":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/a5453bab67b14eb8b01da57f79590409_WREF.pt",
    "SERC":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/95f74a748e2f48d5bed0635228fed41a_SERC.pt",
    "GRSM":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/127191a057e64a7fb7267cc889e56c25_GRSM.pt",
    "DEJU":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/0e3178ac37434aeb90ac207c18a9caf7_DEJU.pt",
    "BONA":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/2975c34a7ca540df9c54d261ef95551e_BONA.pt",
    "TREE":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/f191ef1bb2d54d459eb4f12d73756822_TREE.pt",
    "STEI":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/951f766a48aa4a0baa943d9f6d26f3e0_STEI.pt",
    "UNDE":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/185caf9f910c4fd3a7f5e470b6828090_UNDE.pt",
    "DELA":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/1d22265240bc4e52b77df78ce557e40f_DELA.pt",
    "LENO":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/537b80fd46e64c77a1c367dcbef713e3_LENO.pt",
    "OSBS":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/9fca032e2322479b82506e700de065f5_OSBS.pt",
    "JERC":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/86d51ae4b7a34308bc99c19f8eeadf41_JERC.pt",
    "TALL":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/c52006e0461b4363956335203e42f786_TALL.pt",
    "CLBJ":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/83c1d8fd4c69479185ed3224abb6e8f9_CLBJ.pt",
    "TEAK":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/b9b6882705d24fe6abf12282936ababb_TEAK.pt",
    "SOAP":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/97695ef8ec6a481fb3515d29d2cf33bb_SOAP.pt",
    "YELL":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/e1760a6a288747beb83f055155e49109_YELL.pt",                       
    "MLBS":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/59f4175b26fc48f8b6f16d0598d49411_MLBS.pt",
    "BLAN":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/ce50bb593a484a28b346e7efe357e0fa_BLAN.pt",
    "UKFS":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/1d9305235bf44f319def7915a4b7b21f_UKFS.pt",
    "BART":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/01bfce34aacd455bb1a4b4d80deb16d2_BART.pt",
    "HARV":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/973a01b0c1a349ebaf4fc8454ffc624d_HARV.pt"}

rows = []
for site in species_model_paths:
    try: 
        model_path = species_model_paths[site]
        key = os.path.basename(model_path).split("_")[0]
        api = API()
        experiment = api.get_experiment("bw4sz", "deeptreeattention2", key)    
        overall_micro = experiment.get_metrics("overall_micro")[0]["metricValue"]
        overall_macro = experiment.get_metrics("overall_macro")[0]["metricValue"]
        num_classes = experiment.get_parameters_summary("num_species")["valueCurrent"]
        train_samples = experiment.get_parameters_summary("train_samples")["valueCurrent"]
        test_samples = experiment.get_parameters_summary("test_samples")["valueCurrent"]
        output_row = pd.DataFrame({"site":[site],"Micro-accuracy":[overall_micro],"Macro-accuracy":[overall_macro],"Species":[num_classes],"Train Samples":[train_samples],"Test":[test_samples]})
        rows.append(output_row)
    except:
        print(site)
        continue
    
output_table = pd.concat(rows)
output_table = output_table.sort_values("Micro-accuracy", ascending=False)
output_table.to_csv("/home/b.weinstein/DeepTreeAttention/results/metrics.csv")

    