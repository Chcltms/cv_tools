#!/usr/bin/env python
# coding: utf-8

# In[5]:


import cv2
import os
import matplotlib.pyplot as plt
import glob
import numpy as np
import concurrent.futures
from utils import *
import sys
import concurrent.futures


# In[80]:





# In[127]:


def use_contours(labels_row):
    contours, hierarchy = cv2.findContours(labels_row, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    pre_x = 0
    pre_w = 0
    bbox_list=[]
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        y_end = y + h;x_end = x + w
        bbox_list.append((x, y, x_end, y_end))
    bbox_list.sort()
    bbox_index = 0
    for bbox in bbox_list:
        x, y, x_end, y_end = bbox
        y = int(y*0.8)
        y_end = int(y_end + 0.2*(labels_row.shape[0] - y_end - 1 ))
        '''
        if bbox_index == 0:
            x = int(x*0.8)
            x_end = int(x_end*1.05)
        else:
            x = bbox_list[bbox_index][2] + 0.8(x - bbox_list[bbox_index][2])
            if int(x_end * 1.05) < labels_row.shape[1] - 1:
                x_end = int(x_end * 1.05)
            else:
                x_end = x_end
        '''
        labels_row = cv2.rectangle(labels_row,(x,y),(x_end,y_end),1,cv2.FILLED)
    return labels_row


# In[128]:


def img_save(labels_row,img,raw_shape,path):
    labels = labels_row[:,:,np.newaxis]
    img = (img*labels)
    img = cv2.resize(img, (raw_shape[1],raw_shape[0]))
    cv2.imwrite(background_path+path.split('/')[-1], img)


# In[129]:


def get_result(path):
    img = cv2.imread(path)
    raw_shape = img.shape
    img = cv2.resize(img, (img.shape[1]*2, img.shape[0]*2))
    labels_row = get_mask(img)
    
    labels_row = use_contours(labels_row)
    if labels_row[labels_row.shape[0]-1][labels_row.shape[1]-1] == 1 or labels_row[0][0] == 1:
        labels_row = 1 - labels_row

    #img_background = cv2.inpaint(img_background, labels_row, 1,cv2.INPAINT_NS)
    #cv2.imwrite(background_path+path.split('\\')[-1], img_background)
    #plt.imshow(cv2.drawContours(img, contours, 2, (0,255,0), 3))
    #labels_row = use_contours(labels_row)
    #labels = labels_row[:,:,np.newaxis]
    img = cv2.inpaint(img, labels_row, 1,cv2.INPAINT_NS)
    img = cv2.medianBlur(img,7)
    img = cv2.GaussianBlur(img,(7,7),0)
    #img_save(labels_row,img,raw_shape,path)
    cv2.imwrite(background_path+path.split('/')[-1], img)

# In[104]:

if __name__ == "__main__":
  background_path = 'background_inpaint_original_x/'
  img_list = glob.glob('background_source/*.jpg')
  if not os.path.exists(background_path):
    os.mkdir(background_path)
  with concurrent.futures.ProcessPoolExecutor() as executor:
    executor.map(get_result, img_list)
  #get_result(img_list[0])




