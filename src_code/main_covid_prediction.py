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

# debugger:
from icecream import ic

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
# %% BALANCE TRAINING DATASET -------------------------------- ####
"""
    Since we notice the imbalance in training dataset, let's try random downsampling.
"""
train_pos = TRAIN_DATA_LUT[TRAIN_DATA_LUT["Y"] == 1]
train_neg = TRAIN_DATA_LUT[TRAIN_DATA_LUT["Y"] == 0]
N_balanced = min(len(train_pos), len(train_neg))
# shuffle and resample:
train_pos = train_pos[0:N_balanced]
train_neg = train_neg[0:N_balanced]
NEW_TRAIN_DATA_LUT = pd.concat([train_pos, train_neg])
report_status(data=train_pos, tag="new:train_pos")
report_status(data=train_neg, tag="new:train_neg")
report_status(data=NEW_TRAIN_DATA_LUT, tag="new:train")
TRAIN_DATA_LUT = NEW_TRAIN_DATA_LUT

# %% USER DEFINE: ----- ----- ----- ----- ----- ----- ----- ----- ----- #
@dataclass
class PredictorConfiguration:
    MODEL_TAG            : str              = "default"
    OUT_DIR              : str              = ""
    OUT_DIR_MODELS       : str              = ""
    VERSION              : str              = "default"
    # Settings:
    TOTAL_NUM_EPOCHS     : int              = 5
    LEARNING_RATE        : float            = 0.001
    BATCH_SIZE           : int              = 100
    LOSS_FUNC            : nn               = nn.NLLLoss()
    OPTIMIZER            : optim            = None
    # early stopping:
    EARLY_STOPPING_DECLINE_CRITERION  : int = 5

SELECTED_TARGET = "1LAYER" # <--- select model !!!

# %% INIT: ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- #
### MODEL ###
MODEL_DICT = {
    "1LAYER": {
        "model":
            nn.Sequential(
                # Classifier
                nn.Flatten(1),
                # nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(50176,   2),
                nn.Softmax()
            ),
        "config":
            PredictorConfiguration(
                VERSION="v1",
                OPTIMIZER=optim.SGD,
            ),
    },
    # "VGG11": {
    #     "model":
    #         nn.Sequential(
    #             ## CNN Feature Extraction
    #             nn.Conv2d(  1,  64, 3, 1, 1), nn.BatchNorm2d( 64), nn.ReLU(), nn.MaxPool2d(2,2),
    #             nn.Conv2d( 64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2,2),
    #             nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
    #             nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2,2),
    #             nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
    #             nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2,2),
    #             nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
    #             nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2,2),
    #             # Classifier
    #             nn.Flatten(1),
    #             nn.Linear( 512, 4096), nn.ReLU(), nn.Dropout(0.5),
    #             nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
    #             nn.Linear(4096,   2),
    #         ),
    #     "config":
    #         PredictorConfiguration(VERSION="v1"),
    # },
}


# select model:
SELECTED_NET_MODEL = MODEL_DICT[SELECTED_TARGET]["model"]
SELECTED_NET_CONFIG = MODEL_DICT[SELECTED_TARGET]["config"]
# model specific declaration:
SELECTED_NET_CONFIG.MODEL_TAG = SELECTED_TARGET
SELECTED_NET_CONFIG.OPTIMIZER = SELECTED_NET_CONFIG.OPTIMIZER(
    SELECTED_NET_MODEL.parameters(), lr=SELECTED_NET_CONFIG.LEARNING_RATE
)
### Directory generation ###
OUT_DIR = abspath("output")
MODEL_OUT_DIR = "{}/{}".format(OUT_DIR, SELECTED_TARGET)
SELECTED_NET_CONFIG.OUT_DIR = "{}/{}".format(MODEL_OUT_DIR, SELECTED_NET_CONFIG.VERSION)
SELECTED_NET_CONFIG.OUT_DIR_MODELS = "{}/{}".format(SELECTED_NET_CONFIG.OUT_DIR, "models")
jx_lib.create_folder(DIR=OUT_DIR)
jx_lib.create_folder(DIR=MODEL_OUT_DIR)
jx_lib.create_folder(DIR=SELECTED_NET_CONFIG.OUT_DIR)
jx_lib.create_folder(DIR=SELECTED_NET_CONFIG.OUT_DIR_MODELS)

# define logger:
def _print(content):
    print("[ENGINE] ", content)
    with open(os.path.join(SELECTED_NET_CONFIG.OUT_DIR,"log.txt"), "a") as log_file:
        log_file.write("\n")
        log_file.write("[{}]: {}".format(datetime.datetime.now(), content))

# log model:
_print(str(SELECTED_NET_MODEL))
_print(str(SELECTED_NET_CONFIG))

#%% LOAD NET: ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- #
# check device:
# hardware-acceleration
device = None
if torch.cuda.is_available():
    _print("[ALERT] Attempt to use GPU => CUDA:0")
    device = torch.device("cuda:0")
else:
    _print("[ALERT] GPU not found, use CPU!")
    device = torch.device("cpu")
SELECTED_NET_MODEL.to(device)

# %% LOAD DATASET: ----- ----- ----- ----- ----- ----- ----- ----- #####
# define custom dataset methods:
class CTscanDataSet(Dataset):
    def __init__(self, list_of_img_dir, transform, labels):
        self.list_of_img_dir = list_of_img_dir
        self.transform = transform
        self.labels = labels

    def __len__(self):
        return len(self.list_of_img_dir)

    def __getitem__(self, idx):
        img_loc = self.list_of_img_dir[idx]
        img = Image.open(img_loc).convert('RGB')
        # arr = np.array(img)
        # norm_arr = arr / 255
        # new_img = Image.fromarray(norm_arr.astype('float'),'RGB')
        img_transformed = self.transform(img)
        return (img_transformed, self.labels[idx])
    
    def _report(self):
        N_total = len(self.labels)
        N_pos = np.sum(self.labels)
        N_neg = N_total-N_pos 
        tag = "BALANCED." if N_pos == N_neg else "UNBALANCED !!!"
        return "+: {1}/{0} ({3:.2f}%)  -: {2}/{0} ({4:.2f}%) [{5}]".format(
            N_total, N_pos, N_neg, N_pos/N_total*100, N_neg/N_total*100, tag 
        )

TRANSFORMATION = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    transforms.Grayscale() # apply gray scale
])

# load image:
img_dataset_train = CTscanDataSet(
    list_of_img_dir=TRAIN_DATA_LUT["img_abs_path"], 
    transform=TRANSFORMATION, labels=TRAIN_DATA_LUT["Y"]
)
img_dataset_valid = CTscanDataSet(
    list_of_img_dir=VALID_DATA_LUT["img_abs_path"], 
    transform=TRANSFORMATION, labels=VALID_DATA_LUT["Y"]
)

# Prep. dataloader
train_dataloader = torch.utils.data.DataLoader(
    img_dataset_train, 
    batch_size=SELECTED_NET_CONFIG.BATCH_SIZE, shuffle=True
)
valid_dataloader = torch.utils.data.DataLoader(
    img_dataset_valid, 
    batch_size=SELECTED_NET_CONFIG.BATCH_SIZE, shuffle=True
)

_print("=== Dataset Loaded:")
_print("> Train Dataset: {}".format(train_dataloader.dataset._report()))
_print("> Valid Dataset: {}".format(valid_dataloader.dataset._report()))


# %% PRINT SAMPLE: ----- ----- ----- ----- ----- ----- ---
def plot_sample_from_dataloader(dataloader, tag:str, N_COLS = 5, N_MAX=20):
    N_MAX = min(SELECTED_NET_CONFIG.BATCH_SIZE, N_MAX)
    N_ROWS = int(np.ceil(N_MAX/N_COLS))
    fig, axes = plt.subplots(
        figsize=(N_COLS * 4, N_ROWS * 4), 
        ncols=N_COLS, nrows=N_ROWS
    )
    _print("=== Print Sample Data ({}) [n_display:{} / batch_size:{}]".format(
        tag, N_MAX, SELECTED_NET_CONFIG.BATCH_SIZE))
    # get one batch:
    images, labels = next(iter(dataloader))
    for i in range(N_MAX):
        ax = axes[int(i/N_COLS), i%N_COLS]
        ax.imshow(images[i][0])
        ax.set_title(
            "{}".format(INT_TO_LABEL_LUT[int(labels[i])]),
            color="red" if int(labels[i]) else "blue"
        )
    fig.savefig("{}/plot_{}.png".format(SELECTED_NET_CONFIG.OUT_DIR, tag), bbox_inches = 'tight')

plot_sample_from_dataloader(train_dataloader, tag="training-sample")
plot_sample_from_dataloader(valid_dataloader, tag="validation-sample")

# %% TRAIN: ----- ----- ----- ----- ----- ----- ---
# Reload:
import importlib
importlib.reload(jx_pytorch_lib)
importlib.reload(jx_lib)
from jx_pytorch_lib import ProgressReport, VerboseLevel, CNN_MODEL_TRAINER

# run:
report = CNN_MODEL_TRAINER.train_and_monitor(
    device=device,
    train_dataset=train_dataloader,
    test_dataset=valid_dataloader, 
    optimizer=SELECTED_NET_CONFIG.OPTIMIZER, 
    loss_func=SELECTED_NET_CONFIG.LOSS_FUNC,
    net=SELECTED_NET_MODEL,
    num_epochs=SELECTED_NET_CONFIG.TOTAL_NUM_EPOCHS,
    model_output_path=SELECTED_NET_CONFIG.OUT_DIR_MODELS,
    early_stopping_n_epochs_consecutive_decline=SELECTED_NET_CONFIG.EARLY_STOPPING_DECLINE_CRITERION,
    # max_data_samples=20,
    verbose_level= VerboseLevel.HIGH,
    _print=_print,
)

report.output_progress_plot(
    OUT_DIR=SELECTED_NET_CONFIG.OUT_DIR, 
    tag=SELECTED_NET_CONFIG.VERSION,
    verbose_level=VerboseLevel.HIGH
)
