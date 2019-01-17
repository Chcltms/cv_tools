#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import os
import numpy as np
from sklearn.cluster import KMeans


# In[2]:


def get_mask(img):
    img_cp = np.array(img, dtype = np.float64)/255
    w, h ,d = tuple(img.shape)
    image_array = np.reshape(img_cp, (w * h, d))
    kmeans = KMeans(n_clusters=2,max_iter=500,n_jobs=2).fit(image_array)
    labels_row = np.reshape(kmeans.labels_, (w, h)).astype(np.uint8)
    return labels_row


# In[ ]:




