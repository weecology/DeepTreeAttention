#Extract highly confused classes for visualization

#Visualize neighbor model
from DeepTreeAttention.visualization import extract
from DeepTreeAttention.trees import AttentionModel

mod = AttentionModel(config="/home/b.weinstein/DeepTreeAttention/conf/tree_config.yml")
extract.save_images_to_matlab(mod, classes=["PSMEM","ABLAL","PIEN","ACRU","QURU"], savedir="/orange/idtrees-collab/DeepTreeAttention/examples/")