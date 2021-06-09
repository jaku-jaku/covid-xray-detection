"""
This main would predict covid
"""
# %% Import
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

from ast import literal_eval
import os
import sys
import json

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import re
import emoji
import operator
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import datetime

from PIL import Image

# ML:
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torchvision.models as models

## USER DEFINED:
ABS_PATH = "/home/jx/JXProject/Github/covidx-clubhouse" # Define ur absolute path here

## Custom Files:
def abspath(relative_path):
    return os.path.join(ABS_PATH, relative_path)

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(abspath("src_code"))

import jx_lib
import jx_pytorch_lib


# %% LOAD DATASET INFO: ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- #
LUT_HEADER = ["[patient id]", "[filename]", "[class]", "[data source]"]
# import data
TRAIN_DATA_LUT = pd.read_csv(abspath("data/train.txt"), sep=" ", header=None, names=LUT_HEADER)
VALID_DATA_LUT = pd.read_csv(abspath("data/test.txt"), sep=" ", header=None, names=LUT_HEADER)
# convert class to label 'y'
LABEL_TO_INT_LUT = {
    "positive": 1,
    "negative": 0,
}
INT_TO_LABEL_LUT = {
    1: "positive",
    0: "negative",
}
def class_to_binary(cls):
    return [LABEL_TO_INT_LUT[c] for c in cls]
TRAIN_DATA_LUT["Y"] = class_to_binary(TRAIN_DATA_LUT["[class]"])
VALID_DATA_LUT["Y"] = class_to_binary(VALID_DATA_LUT["[class]"])
# convert filename to absolute path:
def filename_to_abspath(filenames, tag):
    return [abspath("data/{}/{}".format(tag, filename)) for filename in filenames]
TRAIN_DATA_LUT["img_abs_path"] = filename_to_abspath(filenames=TRAIN_DATA_LUT["[filename]"], tag="train")
VALID_DATA_LUT["img_abs_path"] = filename_to_abspath(filenames=VALID_DATA_LUT["[filename]"], tag="test")

# report status:
def report_status(data, tag):
    tp, tn = np.sum(data["Y"] == 1), np.sum(data["Y"] == 0)
    print("{}: +:{}, -:{}".format(tag, tp, tn))

report_status(data=TRAIN_DATA_LUT, tag="train")
report_status(data=VALID_DATA_LUT, tag="valid")
# %% Convert image: ----- ----- ----- ----- ----- -----
import cv2
def img_batch_conversion(PATH_LUT, OUT_DIR):
    jx_lib.create_folder(DIR=OUT_DIR)
    counter = 0
    for img_path, file_name in zip(PATH_LUT["img_abs_path"], PATH_LUT["[filename]"]):
        counter += 1
        print("\r   >[{}/{}]".format(counter,len(PATH_LUT["img_abs_path"])),  end='')
        img = cv2.imread(img_path)
        kernel = np.ones((5, 5), 'uint8')
        img1 = cv2.dilate(img, kernel, iterations=5)
        img2 = cv2.erode(img, kernel, iterations=5)
        # edges = cv.Canny(img,100,200)

        # fig, axes = plt.subplots(figsize=(20, 10), ncols=3)
        # axes[0].imshow(img)
        # axes[1].imshow(img1)
        # axes[2].imshow(img2)

        img_new = np.dstack((img[:,:,0], img1[:,:,1], img2[:,:,2]))
        out_path = "{}/{}".format(OUT_DIR, file_name)
        # plt.imshow(img_new)
        # print(out_path)
        cv2.imwrite(out_path, img_new)
        # break


N_TEST = 400
PATH_LUT_COMP = {
    "[filename]": [ "{}.png".format(i+1) for i in range(N_TEST) ],
    "img_abs_path": [ abspath("data/competition_test/{}.png".format(i+1)) for i in range(N_TEST) ],
}
OUT_DIR = abspath("data/competition_test-custom")
img_batch_conversion(PATH_LUT=PATH_LUT_COMP, OUT_DIR=OUT_DIR)

# OUT_DIR = abspath("data/train-custom")
# img_batch_conversion(PATH_LUT=TRAIN_DATA_LUT, OUT_DIR=OUT_DIR)
# OUT_DIR = abspath("data/valid-custom")
# img_batch_conversion(PATH_LUT=VALID_DATA_LUT, OUT_DIR=OUT_DIR)
# %%
