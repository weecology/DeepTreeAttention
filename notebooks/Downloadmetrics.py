#Download hyperparameters and metrics 
from comet_ml import API
import os
import io
import pandas as pd
import traceback

species_model_paths = {
    "NIWO": "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/000b1ecf0ca6484893e177e3b5d42c7e_NIWO.pt",
    "RMNP": "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/9d71632542494af494c83fb4487747ce_RMNP.pt",    
    "SJER":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/ecfdd5bf772a40cab89e89fa1549f13b_SJER.pt",
    "WREF":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/686204cb0d5343b0b20613a6cf25f69b_WREF.pt",
    "SERC":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/20ce0ca489444e84997e82b4b293e86c_SERC.pt",
    "GRSM":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/50da72a1cb6042338d96244d968a365b_GRSM.pt",
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
    "TEAK":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/ca17bf7c36fe42e6bd83a358243c012b_TEAK.pt",
    "SOAP":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/c56b937a94f84e9da774370f4e46a110_SOAP.pt",
    "YELL":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/f2c069f59b164795af482333a5e7fffb_YELL.pt",                       
    "MLBS":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/b5efc0037529431092db587727fb4fe9_MLBS.pt",
    "BLAN":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/533e410797c945618c72b2a54176ed61_BLAN.pt",
    "UKFS":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/084b83c44d714f23b9d96e0a212f11f1_UKFS.pt",
    "BART":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/58ac69d485d645ad8b4a872ff7ea7588_BART.pt",
    "HARV":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/9130a6b5ce544e1280283bf60cab63b0_HARV.pt"}

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
        metrics = pd.DataFrame(experiment.get_metrics_summary())
        metrics.to_csv("/home/b.weinstein/DeepTreeAttention/results/{}_metrics.csv".format(site))
        train_asset = experiment.get_asset_list()
        asset_train_id = [x for x in train_asset if "train.csv" == x["fileName"]][0]["assetId"]
        train_csv = experiment.get_asset(asset_train_id).decode()
        train_df = pd.read_csv(io.StringIO(train_csv))
        train_df.to_csv("/home/b.weinstein/DeepTreeAttention/results/{}_train.csv".format(site))

        output_row = pd.DataFrame({"site":[site],"Micro-accuracy":[overall_micro],"Macro-accuracy":[overall_macro],"Species":[num_classes],"Train Samples":[train_samples],"Test":[test_samples]})
        rows.append(output_row)
    except:
        traceback.print_exc()
        print(site)
        continue
    
output_table = pd.concat(rows)
output_table = output_table.sort_values("Micro-accuracy", ascending=False)
output_table.to_csv("/home/b.weinstein/DeepTreeAttention/results/metrics.csv")

    