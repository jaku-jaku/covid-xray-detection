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
    BATCH_SIZE           : int              = 1000
    LOSS_FUNC            : nn               = nn.NLLLoss()
    OPTIMIZER            : optim            = None
    # early stopping:
    EARLY_STOPPING_DECLINE_CRITERION  : int = 5

# SELECTED_TARGET = "1LAYER" # <--- select model !!!
SELECTED_TARGET = "CUSTOM-MODEL" # <--- select model !!!
PRINT_SAMPLES = False

# %% MODEL:
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
class ResNet(nn.Module):
    
    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 , num_classes)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None  

        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        
        self.inplanes = planes
        
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        x = self.conv1(x)           # 224x224
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)         # 112x112

        x = self.layer1(x)          # 56x56
        x = self.layer2(x)          # 28x28
        x = self.layer3(x)          # 14x14
        x = self.layer4(x)          # 7x7

        x = self.avgpool(x)         # 1x1
        x = torch.flatten(x, 1)     # remove 1 X 1 grid and make vector of tensor shape 
        x = self.fc(x)

        return x

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
                BATCH_SIZE=1000,
            ),
        "transformation":
            transforms.Compose([
                # transforms.RandomHorizontalFlip(p=0.5),
                # transforms.RandomVerticalFlip(p=0.5),
                # transforms.RandomPerspective(),
                # transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
                transforms.Resize(255),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5)),
                transforms.Grayscale() # apply gray scale
            ]),
    },
    "4LAYER": {
        "model":
            nn.Sequential(
                # Classifier
                nn.Flatten(1),
                nn.Linear(50176, 1000),
                nn.Linear(1000, 1000),
                nn.Linear(1000, 100),
                nn.Linear(100,   2),
                nn.Softmax()
            ),
        "config":
            PredictorConfiguration(
                VERSION="v1",
                OPTIMIZER=optim.SGD,
                BATCH_SIZE=200,
                TOTAL_NUM_EPOCHS=20
            ),
        "transformation":
            transforms.Compose([
                # transforms.RandomHorizontalFlip(p=0.5),
                # transforms.RandomVerticalFlip(p=0.5),
                # transforms.RandomPerspective(),
                # transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
                transforms.Resize(255),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5)),
                transforms.Grayscale() # apply gray scale
            ]),
    },
    "CUSTOM-MODEL": {
        "model":
            nn.Sequential(
                # Feature Extraction:
                ResNet(BasicBlock, [3, 4, 6, 3], num_classes=2), # ResNet 34
                # # Classifier:
                # nn.ReLU(), 
                # # nn.Dropout(0.2), 
                # nn.Linear(1000, 2), 
                nn.Softmax(),
            ),
            # models.inception_v3(),
        "config":
            PredictorConfiguration(
                VERSION="v6",
                OPTIMIZER=optim.SGD,
                LEARNING_RATE=0.03,
                BATCH_SIZE=20,
                TOTAL_NUM_EPOCHS=20,
                EARLY_STOPPING_DECLINE_CRITERION=5,
            ),
        "transformation":
            transforms.Compose([
                # v3
                # transforms.RandomHorizontalFlip(p=0.5),
                # transforms.RandomVerticalFlip(p=0.5),
                # transforms.RandomPerspective(),
                # transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
                # same:
                transforms.Resize(300),
                transforms.CenterCrop(300),
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5)),
                # transforms.Grayscale() # apply gray scale
            ]),
    },
}

# # Classifier:
# MODEL_DICT["CUSTOM-MODEL"]["model"].fc = nn.Sequential(
#     nn.Linear(2048, 512),
#     nn.ReLU(),
#     nn.Dropout(0.2),
#     nn.Linear(512, 2),
#     nn.LogSoftmax(dim=1)
# )

# select model:
SELECTED_NET_MODEL = MODEL_DICT[SELECTED_TARGET]["model"]
SELECTED_NET_CONFIG = MODEL_DICT[SELECTED_TARGET]["config"]
SELECTED_NET_TRANSFORMATION = MODEL_DICT[SELECTED_TARGET]["transformation"]
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

# load image:
img_dataset_train = CTscanDataSet(
    list_of_img_dir=TRAIN_DATA_LUT["img_abs_path"], 
    transform=SELECTED_NET_TRANSFORMATION, labels=TRAIN_DATA_LUT["Y"]
)
img_dataset_valid = CTscanDataSet(
    list_of_img_dir=VALID_DATA_LUT["img_abs_path"], 
    transform=SELECTED_NET_TRANSFORMATION, labels=VALID_DATA_LUT["Y"]
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
def plot_sample_from_dataloader(dataloader, tag:str, N_COLS = 4, N_MAX=20):
    N_MAX = min(SELECTED_NET_CONFIG.BATCH_SIZE, N_MAX)
    N_COLS = min(N_COLS, N_MAX)
    N_ROWS = int(np.ceil(N_MAX/N_COLS)) * 2
    fig, axes = plt.subplots(
        figsize=(N_COLS * 8, N_ROWS * 8), 
        ncols=N_COLS, nrows=N_ROWS
    )
    _print("=== Print Sample Data ({}) [n_display:{} / batch_size:{}]".format(
        tag, N_MAX, SELECTED_NET_CONFIG.BATCH_SIZE))
    # get one batch:
    images, labels = next(iter(dataloader))
    for i in range(N_MAX):
        print("\r   >[{}/{}]".format(i+1,N_MAX),  end='')
        # Plot img:
        id_ = i * 2
        ax = axes[int(id_/N_COLS), id_%N_COLS]
        # show remapped image, since the range was distorted by normalization
        ax.imshow((np.dstack((images[i][0], images[i][1], images[i][2])) + 1)/2, vmin=0, vmax=1)
        ax.set_title(
            "{}".format(INT_TO_LABEL_LUT[int(labels[i])]),
            color="red" if int(labels[i]) else "blue"
        )

        # Plot Hist:
        id_ += 1
        ax = axes[int(id_/N_COLS), id_%N_COLS]
        ax.hist(np.ravel(images[i][0]), bins=256, color='r', alpha = 0.5, range=[-1, 1])
        ax.hist(np.ravel(images[i][1]), bins=256, color='g', alpha = 0.5, range=[-1, 1])
        ax.hist(np.ravel(images[i][2]), bins=256, color='b', alpha = 0.5, range=[-1, 1])
        ax.legend(['R', 'G', 'B'])
        ax.set_title("<- Histogram")
        
    fig.savefig("{}/plot_{}.png".format(SELECTED_NET_CONFIG.OUT_DIR, tag), bbox_inches = 'tight')

if PRINT_SAMPLES:
    plot_sample_from_dataloader(train_dataloader, tag="training-sample")
    plot_sample_from_dataloader(valid_dataloader, tag="validation-sample")


# # %% Playground
# images, labels = next(iter(train_dataloader))
# img = images[0]

# # %% T
# fig, axes = plt.subplots(figsize=(20, 12), ncols=3, nrows=2)
# axes[0,0].imshow(np.dstack((img[0], img[0], img[0])))
# axes[0,1].imshow(np.dstack((img[1], img[1], img[1])))
# axes[0,2].imshow(np.dstack((img[2], img[2], img[2])))

# # do sth:
# import cv2

# kernel = np.ones((5, 5), 'uint8')
# img2 = cv2.dilate(img.numpy(), kernel, iterations=10)
# img3 = cv2.erode(img.numpy(), kernel, iterations=10)
# img4 = cv2.morphologyEx(img.numpy(), cv2.MORPH_OPEN, kernel)
# axes[1,0].imshow(np.dstack((img2[0], img2[0], img2[0])))
# axes[1,1].imshow(np.dstack((img3[1], img3[1], img3[1])))
# axes[1,2].imshow(np.dstack((img4[2], img4[2], img4[2])))
# axes[1,0].set_title("dilation")
# axes[1,1].set_title("erosion")
# axes[1,2].set_title("morpho-open")


# %% TRAIN: ----- ----- ----- ----- ----- ----- ---
# Reload:
import importlib
importlib.reload(jx_pytorch_lib)
importlib.reload(jx_lib)
from jx_pytorch_lib import ProgressReport, VerboseLevel, CNN_MODEL_TRAINER

# run:
report, best_net = CNN_MODEL_TRAINER.train_and_monitor(
    device=device,
    train_dataset=train_dataloader,
    test_dataset=valid_dataloader, 
    optimizer=SELECTED_NET_CONFIG.OPTIMIZER, 
    loss_func=SELECTED_NET_CONFIG.LOSS_FUNC,
    net=SELECTED_NET_MODEL,
    num_epochs=SELECTED_NET_CONFIG.TOTAL_NUM_EPOCHS,
    model_output_path=SELECTED_NET_CONFIG.OUT_DIR_MODELS,
    target_names=LABEL_TO_INT_LUT,
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

# %% EVALUATE WITH COMPETITION TEST DATASET -------------------------------- %%
from sklearn.metrics import confusion_matrix, classification_report
def evaluate_net(net, dataloader):
    _print("> Evaluation Begin ...")
    y_true = []
    y_pred = []
    for X, y in dataloader:
        if device != None:
            X = X.to(device)
            y = y.to(device)

        # Predict:
        y_prediction = net(X)

        # record:
        y_true.extend(y.cpu().detach().numpy())
        y_pred.extend(y_prediction.argmax(dim=1).cpu().detach().numpy())
    _print("> Evaluation Complete")
    return y_true, y_pred

def eval_competition(net, dataloader, tag):
    # eval:
    y_true, y_pred = evaluate_net(net=net, dataloader=dataloader)
    _print("> {3} +:{1}/{0} 1:{2}/{0}".format(N_TEST, np.sum(y_pred), N_TEST-np.sum(y_pred), tag))

    OUT_FILE_PATH = "{}/y_pred[{}].txt".format(SELECTED_NET_CONFIG.OUT_DIR, tag)
    with open(OUT_FILE_PATH, "w") as file_out:
        file_out.write("\n".join(["{}".format(yi) for yi in y_pred]))

# form dataloader:
N_TEST = 400
COMPETITION_PATH = [ abspath("data/competition_test/{}.png".format(i+1)) for i in range(N_TEST) ]
COMPETITION_Y = np.zeros(N_TEST)
img_dataset_competition = CTscanDataSet(
    list_of_img_dir=COMPETITION_PATH, 
    transform=SELECTED_NET_TRANSFORMATION, labels=COMPETITION_Y
)
competition_dataloader = torch.utils.data.DataLoader(
    img_dataset_competition, 
    batch_size=SELECTED_NET_CONFIG.BATCH_SIZE, shuffle=False
)

_print("=== Dataset Loaded:")
_print("> Competition Dataset: {}".format(competition_dataloader.dataset._report()))

#eval:
eval_competition(net=best_net, dataloader=competition_dataloader, tag="best")
eval_competition(net=SELECTED_NET_MODEL, dataloader=competition_dataloader, tag="final")
