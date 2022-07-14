from src import start_cluster
client = start_cluster.start(cpus=25)

for x in range(10):
    wrapper(iteration=x, client=client, savedir="/blue/ewhite/b.weinstein/DeepTreeAttention/results/06ee8e987b014a4d9b6b824ad6d28d83")