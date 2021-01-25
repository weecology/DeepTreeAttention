#Extract highly confused classes for visualization

#Visualize neighbor model
from DeepTreeAttention.visualization import extract
from DeepTreeAttention.trees import AttentionModel

mod = AttentionModel()
extract.save_images_to_matlab(mod, classes=["ABLAL","PIEN"], savedir="/orange/idtrees-collab/DeepTreeAttention/examples/")