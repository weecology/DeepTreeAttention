from src.multonomial import wrapper
from src import start_cluster

client = start_cluster.start(cpus=50, mem_size="5GB")
for x in range(2):
    wrapper(client=client, iteration=x, experiment_key="209ca047ed004d778c0f0e728e126bda")

