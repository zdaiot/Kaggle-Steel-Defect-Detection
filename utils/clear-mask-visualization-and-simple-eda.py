#!/usr/bin/env python
# coding: utf-8

# # Let's Detect Steel Defect!

# ## import modules and define models

# In[1]:


import numpy as np # linear algebra
import pandas as pd
pd.set_option("display.max_rows", 101)
import os
print('Current work path: ', os.getcwd())
print(os.listdir("datasets/Steel_data"))
import cv2
import json
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
# plt.rcParams["font.size"] = 15
from collections import Counter
from PIL import Image
import math
import seaborn as sns
from collections import defaultdict
from pathlib import Path
import cv2
from tqdm import tqdm


# In[2]:


input_dir = "datasets/Steel_data/"


# ## read all text data
# #### file description
# * train_images/ - folder of training images
# * test_images/ - folder of test images (you are segmenting and classifying these images)
# * train.csv - training annotations which provide segments for defects (ClassId = [1, 2, 3, 4])
# * sample_submission.csv - a sample submission file in the correct format; note, **each ImageId 4 rows, one for each of the 4 defect classes**

# In[3]:


train_df = pd.read_csv("datasets/Steel_data/train.csv")
sample_df = pd.read_csv("datasets/Steel_data/sample_submission.csv")


# In[4]:


train_df.head()


# In[5]:


sample_df.head()


# ### First, check the number of each class.

# In[6]:


class_dict = defaultdict(int)

kind_class_dict = defaultdict(int)

no_defects_num = 0
defects_num = 0

for col in range(0, len(train_df), 4):
    img_names = [str(i).split("_")[0] for i in train_df.iloc[col:col+4, 0].values]
    if not (img_names[0] == img_names[1] == img_names[2] == img_names[3]):
        raise ValueError
        
    labels = train_df.iloc[col:col+4, 1]
    if labels.isna().all():
        no_defects_num += 1
    else:
        defects_num += 1
    
    kind_class_dict[sum(labels.isna().values == False)] += 1
        
    for idx, label in enumerate(labels.isna().values.tolist()):
        if label == False:
            class_dict[idx+1] += 1


# In[7]:


print("the number of images with no defects: {}".format(no_defects_num))
print("the number of images with defects: {}".format(defects_num))


# In[8]:


fig, ax = plt.subplots()
sns.barplot(x=list(class_dict.keys()), y=list(class_dict.values()), ax=ax)
ax.set_title("the number of images for each class")
ax.set_xlabel("class")
class_dict


# * There are similar numbers of images with and without defects.
# * class is imbalanced

# ### How many classes do each image have?

# In[9]:


fig, ax = plt.subplots()
sns.barplot(x=list(kind_class_dict.keys()), y=list(kind_class_dict.values()), ax=ax)
ax.set_title("Number of classes included in each image")
ax.set_xlabel("number of classes in the image")
kind_class_dict


# * almost image have no defect or one kind of defect

# ## check image data
# ### image size

# In[10]:


train_size_dict = defaultdict(int)
train_path = Path("datasets/Steel_data/train_images/")

for img_name in train_path.iterdir():
    img = Image.open(img_name)
    train_size_dict[img.size] += 1


# In[11]:


train_size_dict


# In[12]:


test_size_dict = defaultdict(int)
test_path = Path("datasets/Steel_data/test_images/")

for img_name in test_path.iterdir():
    img = Image.open(img_name)
    test_size_dict[img.size] += 1


# In[13]:


test_size_dict


# * All image have same shape, (1600, 256).

# # Let's visualization masks!

# In[14]:


palet = [(249, 192, 12), (0, 185, 241), (114, 0, 218), (249,50,12)]


# In[15]:


def name_and_mask(start_idx):
    col = start_idx
    img_names = [str(i).split("_")[0] for i in train_df.iloc[col:col+4, 0].values]
    if not (img_names[0] == img_names[1] == img_names[2] == img_names[3]):
        raise ValueError

    labels = train_df.iloc[col:col+4, 1]
    mask = np.zeros((256, 1600, 4), dtype=np.uint8)

    for idx, label in enumerate(labels.values):
        if label is not np.nan:
            mask_label = np.zeros(1600*256, dtype=np.uint8)
            label = label.split(" ")
            positions = map(int, label[0::2])
            length = map(int, label[1::2])
            for pos, le in zip(positions, length):
                mask_label[pos-1:pos+le-1] = 1
            mask[:, :, idx] = mask_label.reshape(256, 1600, order='F')
    return img_names[0], mask


# In[16]:


def show_mask_image(col):
    name, mask = name_and_mask(col)
    img = cv2.imread(str(train_path / name))
    fig, ax = plt.subplots(figsize=(15, 15))

    for ch in range(4):
        contours, _ = cv2.findContours(mask[:, :, ch], cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for i in range(0, len(contours)):
            cv2.polylines(img, contours[i], True, palet[ch], 2)
    ax.set_title(name)
    ax.imshow(img)
    plt.show()


# In[17]:


fig, ax = plt.subplots(1, 4, figsize=(15, 5))
for i in range(4):
    ax[i].axis('off')
    ax[i].imshow(np.ones((50, 50, 3), dtype=np.uint8) * palet[i])
    ax[i].set_title("class color: {}".format(i+1))
fig.suptitle("each class colors")

plt.show()


# In[18]:


idx_no_defect = []
idx_class_1 = []
idx_class_2 = []
idx_class_3 = []
idx_class_4 = []
idx_class_multi = []
idx_class_triple = []

for col in range(0, len(train_df), 4):
    img_names = [str(i).split("_")[0] for i in train_df.iloc[col:col+4, 0].values]
    if not (img_names[0] == img_names[1] == img_names[2] == img_names[3]):
        raise ValueError
        
    labels = train_df.iloc[col:col+4, 1]
    if labels.isna().all():
        idx_no_defect.append(col)
    elif (labels.isna() == [False, True, True, True]).all():
        idx_class_1.append(col)
    elif (labels.isna() == [True, False, True, True]).all():
        idx_class_2.append(col)
    elif (labels.isna() == [True, True, False, True]).all():
        idx_class_3.append(col)
    elif (labels.isna() == [True, True, True, False]).all():
        idx_class_4.append(col)
    elif labels.isna().sum() == 1:
        idx_class_triple.append(col)
    else:
        idx_class_multi.append(col)


# ## images with no defect

# In[19]:


for idx in idx_no_defect[:5]:
    show_mask_image(idx)


# ## images with defect(label: 1)

# In[20]:


for idx in idx_class_1[:5]:
    show_mask_image(idx)


# ## images with defect(label: 2)

# In[21]:


for idx in idx_class_2[:5]:
    show_mask_image(idx)


# ## images with defect(label: 3)

# In[22]:


for idx in idx_class_3[:5]:
    show_mask_image(idx)


# ## images with defect(label: 4)

# In[23]:


for idx in idx_class_4[:5]:
    show_mask_image(idx)


# ## images with defect(contain multi label)

# In[24]:


for idx in idx_class_multi[:5]:
    show_mask_image(idx)


# ## images with defect(contain 3 type label)

# In[25]:


for idx in idx_class_triple:
    show_mask_image(idx)


# * We can see 4 type defect

# # Is there the pixel that have multi label?

# In[26]:


for col in tqdm(range(0, len(train_df), 4)):
    name, mask = name_and_mask(col)
    if (mask.sum(axis=2) >= 2).any():
        show_mask_image(col)


# * All pixels have 1 or less label.

# # Thank you very much for reading my post through to the end.
# Please tell me when I make mistakes in program and English.  
# I hope this kernel will help.  
# If you think this kernel is useful, please upvote.  

# In[ ]:




