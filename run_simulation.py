import comet_ml
from src import data
from src import simulation
import pandas as pd
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
resultdf = []
config = data.read_config("simulation.yml")
for x in range(1):
    result = simulation.run(ID=x, config=config)
    resultdf.append(result)
    
resultdf = pd.concat(resultdf)
resultdf.to_csv("data/processed/simulation.csv")