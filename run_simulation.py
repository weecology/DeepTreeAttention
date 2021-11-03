from src import start_cluster
from src import data
from src import simulation
from distributed import wait
import pandas as pd

client = start_cluster.start(gpus=2)
futures = []
config = data.read_config("simulation.yml")
for x in range(10):
    future = client.submit(simulation.run, ID=x, config=config)
    futures.append(future)
wait(futures)
resultdf = []
for x in futures:
    result = x.result()
    resultdf.append(result)

resultdf = pd.concat(resultdf)
resultdf.to_csv("data/processed/simulation.csv")