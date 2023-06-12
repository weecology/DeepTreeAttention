import geopandas as gpd
import glob
import os
import pandas as pd

species_model_paths = {
    "NIWO": "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/4a962f4745204a82b3688ed505cd76d8_['NIWO', 'RMNP'].pt",
    "SJER":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/87138a0b383c4dfea2df8fb3d6e48119_['SJER'].pt",
    "MOAB":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/54db5e883404420a95af36787f4395d3_['MOAB', 'REDB'].pt",
    "WREF":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/f0f2e7eb0e33484dadcfa011bc6ac745_['WREF'].pt",
    "REDB":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/54db5e883404420a95af36787f4395d3_['MOAB', 'REDB'].pt",
    "SERC":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/920a0d718f894963a961437622be3a97_['SERC', 'GRSM'].pt",
    "GRSM":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/920a0d718f894963a961437622be3a97_['SERC', 'GRSM'].pt",
    "DEJU":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/a86cdf52b3d14568b2d7574a13185868_['BONA', 'DEJU'].pt",
    "BONA":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/a86cdf52b3d14568b2d7574a13185868_['BONA', 'DEJU'].pt",
    "TREE":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/6efefee06802491f897e830d6fc3b19e_['TREE', 'STEI', 'UNDE'].pt",
    "STEI":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/6efefee06802491f897e830d6fc3b19e_['TREE', 'STEI', 'UNDE'].pt",
    "UNDE":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/6efefee06802491f897e830d6fc3b19e_['TREE', 'STEI', 'UNDE'].pt",
    "DELA":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/f418662238b84ef383f852c0821eab4b_['DELA', 'LENO'].pt",
    "LENO":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/f418662238b84ef383f852c0821eab4b_['DELA', 'LENO'].pt",
    "OSBS":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/25bb11c702b64988b7c8258ac0126f02_['OSBS', 'JERC', 'TALL', 'DSNY'].pt",
    "JERC":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/25bb11c702b64988b7c8258ac0126f02_['OSBS', 'JERC', 'TALL', 'DSNY'].pt",
    "TALL":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/25bb11c702b64988b7c8258ac0126f02_['OSBS', 'JERC', 'TALL', 'DSNY'].pt",
    "DSNY":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/25bb11c702b64988b7c8258ac0126f02_['OSBS', 'JERC', 'TALL', 'DSNY'].pt",
    "CLBJ":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/42846206ea4e403c9cdb4ba809f1097e_['CLBJ', 'KONZ'].pt",
    "TEAK":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/8b2940d920ee48b2ac47adf462fc99a6_['TEAK', 'SOAP', 'YELL', 'ABBY'].pt",
    "SOAP":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/8b2940d920ee48b2ac47adf462fc99a6_['TEAK', 'SOAP', 'YELL', 'ABBY'].pt",
    "YELL":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/8b2940d920ee48b2ac47adf462fc99a6_['TEAK', 'SOAP', 'YELL', 'ABBY'].pt",                       
    "MLBS":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/9af9ba5a9e1148daa365d3c893cde875_['MLBS','BLAN','SCBI','UKFS'].pt",
    "BLAN":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/9af9ba5a9e1148daa365d3c893cde875_['MLBS','BLAN','SCBI','UKFS'].pt",
    "SCBI":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/9af9ba5a9e1148daa365d3c893cde875_['MLBS','BLAN','SCBI','UKFS'].pt",
    "UKFS":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/9af9ba5a9e1148daa365d3c893cde875_['MLBS','BLAN','SCBI','UKFS'].pt",
    "BART":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/9821d98c5b474b04bf41edbf0d3d4d96_['BART', 'HARV'].pt",
    "HARV":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/9821d98c5b474b04bf41edbf0d3d4d96_['BART', 'HARV'].pt"}

for model in species_model_paths:
    prediction_dir = os.path.join("/blue/ewhite/b.weinstein/DeepTreeAttention/results/",
                                      os.path.splitext(os.path.basename(model))[0])  
    files = glob.glob("{}/*.shp".format(prediction_dir))
    total_counts = pd.Series()
    for f in files:
        ser = gpd.read_file(f)[["ensembleTa"]].value_counts().reset_index()      
        total_counts = total_counts.add(ser, fill_value=0)
    
    total_counts.to_csv("{}/abundance.csv".format(prediction_dir))
    
        