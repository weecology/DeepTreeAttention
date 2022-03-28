import rasterio as rio
import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt


raw_image = rio.open("notebooks/data/OSBS_IFAS.contrib.105_hyperspectral.tif").read()
raw_image.shape

#Normalize
data = raw_image.reshape(raw_image.shape[0], np.prod(raw_image.shape[1:])).T
data  = preprocessing.scale(data)
raw_image_norm = data.reshape(raw_image.shape)

#Plot each pixel and the mean
for x in raw_image.reshape(raw_image.shape[0], np.prod(raw_image.shape[1:])).T:
    plt.plot(x)
plt.plot(raw_image.mean(axis=(1,2)))

#Plot each pixel and the mean
plt.figure()
for x in raw_image_norm.reshape(np.prod(raw_image_norm.shape[1:]), raw_image_norm.shape[0]):
    plt.plot(x)
plt.plot(raw_image_norm.mean(axis=(1,2)))


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