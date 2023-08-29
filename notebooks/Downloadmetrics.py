#Download hyperparameters and metrics 
from comet_ml import API
import os
import io
import pandas as pd
import traceback
from src.model_list import species_model_paths

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

    