import rasterio as rio
import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt


a2021 = rio.open("notebooks/data/OSBS_graves.contrib.584_2021.tif").read()
a2019 = rio.open("notebooks/data/OSBS_graves.contrib.584_2019.tif").read()
a2018 = rio.open("notebooks/data/OSBS_graves.contrib.584_2018.tif").read()

#Normalize
data = a2021.reshape(np.prod(a2021.shape[1:]), a2021.shape[0])
data  = preprocessing.minmax_scale(data)
a2021 = data.reshape(a2021.shape)
a2021 = a2021.reshape(-1, a2021.shape[0])

data = a2018.reshape(a2018.shape[0], np.prod(a2018.shape[1:]))
data  = preprocessing.minmax_scale(data)
a2018 = data.reshape(a2018.shape)
a2018 = a2018.reshape(-1, a2018.shape[0])

data = a2019.reshape(a2019.shape[0], np.prod(a2019.shape[1:]))
data  = preprocessing.minmax_scale(data)
a2019 = data.reshape(a2019.shape)
a2019 = a2019.reshape(-1, a2019.shape[0])

fig, ax = plt.subplots(3,1 )
ax[0].plot(a2019.mean(axis=0), color="red")
ax[1].plot(a2021.mean(axis=0), color="purple")
ax[2].plot(a2018.mean(axis=0), color="black")
plt.show()

fig, ax = plt.subplots(3,1 )
ax[0].plot(a2019.min(axis=0), color="red")
ax[1].plot(a2021.min(axis=0), color="purple")
ax[2].plot(a2018.min(axis=0), color="black")
ax[0].plot(a2019.max(axis=0), color="red")
ax[1].plot(a2021.max(axis=0), color="purple")
ax[2].plot(a2018.max(axis=0), color="black")
plt.show()



fig = plt.Figure()
plt.plot(a2019.min(axis=0), color="red")
plt.plot(a2019.max(axis=0), color="red")
plt.plot(a2021.min(axis=0), color="brown")
plt.plot(a2021.max(axis=0), color="brown")
plt.plot(a2018.min(axis=0), color="purple")
plt.plot(a2018.max(axis=0), color="purple")
plt.show()


#At the tile level
tile2018 = rio.open("/Users/benweinstein/Downloads/2021_OSBS_6_404000_3285000_image_hyperspectral_2018.tif").read()
tile2019 = rio.open("/Users/benweinstein/Downloads/2021_OSBS_6_404000_3285000_image_hyperspectral_2019.tif").read()

tile2018[tile2018 > 10000] = 0

fig = plt.Figure()
plt.plot(tile2018.min(axis=(1,2)), color="red")
plt.plot(tile2018.max(axis=(1,2)), color="black")
plt.plot(tile2018.mean(axis=(1,2)), color="purple")
plt.show()

fig = plt.Figure()
plt.plot(tile2019.min(axis=(1,2)), color="red")
plt.plot(tile2019.max(axis=(1,2)), color="black")
plt.plot(tile2019.mean(axis=(1,2)), color="purple")
plt.show()

#normalize
data = tile2018.reshape(tile2018.shape[0], np.prod(tile2018.shape[1:]))
data  = preprocessing.scale(data)
tile2018norm = data.reshape(tile2018.shape)
tile2018norm = tile2018.reshape(-1, a2019.shape[0])


fig = plt.Figure()
plt.plot(tile2018norm.min(axis=0), color="red")
plt.plot(tile2018norm.max(axis=0, color="black")
plt.plot(tile2018norm.mean(axis=0), color="purple")
plt.show()