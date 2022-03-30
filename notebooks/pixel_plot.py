import rasterio as rio
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np

raw_image = rio.open("notebooks/data/OSBS_IFAS.contrib.105_hyperspectral.tif").read()
for x in raw_image.reshape(raw_image.shape[0], np.prod(raw_image.shape[1:])).T:
    plt.plot(x)
plt.plot(raw_image.mean(axis=(1,2)))

np.testing.assert_almost_equal(raw_image.reshape(369,12).T[0], raw_image[:,0,0])
data = raw_image.reshape(raw_image.shape[0], np.prod(raw_image.shape[1:])).T
norm_data = preprocessing.minmax_scale(data,axis=1).T
raw_image_norm = norm_data.reshape(raw_image.shape)
plt.close('all')

plt.Figure()
for x in raw_image_norm.reshape(raw_image.shape[0], np.prod(raw_image.shape[1:])).T:
    plt.plot(x)
plt.show()



