from deepforest import main
import numpy as np
import rasterio as rio
from matplotlib import pyplot as plt

m = main.deepforest()
m.use_release()
m.model.nms_thresh = 0.1

src=rio.open("/orange/idtrees-collab/NeonTreeEvaluation/evaluation/RGB/MLBS_020_2018.tif")
img = src.read()
img = np.rollaxis(img, 0, 3)
boxes=m.predict_image(img, return_plot=True)
plt.imshow(boxes[:,:,::-1])
plt.show()
