"""
This main would evaluate covid for competition organizer use
"""


"""
INPUT:
    1. The first input argument receives a list of test images paths. The format of input is similar to this:
        ["/user/home/data/img1.png", "/user/home/data/img2.png", ... ]
    2. The second input argument is the path to the pretrained model that is used for your model to make prediction. 
OUTPUT:
    > outputs a list containing numbers 1 and 0 based on if the corresponding image is Covid positive or not respectively
        [1,0,...]
"""
# %% Import
# Python:
import enum
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from typing import List
import sys, os
local_module_path = os.path.dirname(os.path.abspath(__file__))
if local_module_path not in sys.path:
    sys.path.append(local_module_path)
from PIL import Image

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Classical CV:
import cv2

# CNN modules:
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset

import jx_lib

from tool_data_gen import img_batch_conversion, abs_data_path, LUT_HEADER, class_to_binary, filename_to_abspath, report_status, LABEL_TO_INT_LUT

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

CUSTOM_MODEL_DICT = {
    "model":
        nn.Sequential(
            # Feature Extraction:
            # ResNet(BasicBlock, [0,1,1,1], num_classes=2), # ResNet reduced v9 - ResNet10 - ablation
            # ResNet(BasicBlock, [1,1,1,1], num_classes=2), # ResNet reduced v8 - ResNet10
            # ResNet(BasicBlock, [1,2,3,2], num_classes=2), # ResNet reduced v7
            ResNet(BasicBlock, [3, 4, 6, 3], num_classes=2), # ResNet 34
            # Classifier:
            nn.Softmax(dim=1),
        ),
    "transformation":
        transforms.Compose([
            # same:
            transforms.CenterCrop(320),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5)),
        ]),
}
class CTscanDataSet(Dataset):
    def __init__(self, list_of_img_dir, transform):
        self.list_of_img_dir = list_of_img_dir
        self.transform = transform

    def __len__(self):
        return len(self.list_of_img_dir)

    def __getitem__(self, idx):
        img_loc = self.list_of_img_dir[idx]
        img = Image.open(img_loc).convert('RGB')
        # arr = np.array(img)
        # norm_arr = arr / 255
        # new_img = Image.fromarray(norm_arr.astype('float'),'RGB')
        img_transformed = self.transform(img)
        return img_transformed
    

# %% Evaluation Script
def eval(
    list_of_images  :List[str], 
    model_path      :str, 
    network_arch,
    size            :tuple = (320,320),
    model_dict_type :bool = False, # if saved as model dictionary
    folder_name_for_pre_process_cache: str = "_reduced",
    PROCESSING_NEEDED:bool = True
) -> List[int]:
    # let's pre-process images first:
    print("[Pre-processing] image conversion:")
    PATH_LUT = {}
    PATH_LUT["img_abs_path"]=list_of_images
    PATH_LUT["[filename]"]=[ os.path.basename(path) for path in list_of_images ]
    if PROCESSING_NEEDED: 
        OUT_DIR = os.path.dirname(list_of_images[0]) + folder_name_for_pre_process_cache
        LIST_REDUCED_IMG_PATH = img_batch_conversion(PATH_LUT=PATH_LUT, OUT_DIR=OUT_DIR, size=(320,320), POST_HOMOGRAPHY=True)
    else:
        LIST_REDUCED_IMG_PATH =  PATH_LUT["img_abs_path"]
    
    # load models:
    if model_dict_type:
        trained_net = network_arch
        trained_net.load_state_dict(torch.load(model_path, map_location="cuda:0"))
    else:
        trained_net = torch.load(model_path)

    # hardware-acceleration
    print("\n[Preparation] Prepare Model:")
    device = None
    if torch.cuda.is_available():
        print("[ALERT] Attempt to use GPU => CUDA:0")
        device = torch.device("cuda:0")
    else:
        print("[ALERT] GPU not found, use CPU!")
        device = torch.device("cpu")
    
    trained_net.to(device)

    img_dataset_ = CTscanDataSet(
        list_of_img_dir=LIST_REDUCED_IMG_PATH, 
        transform=CUSTOM_MODEL_DICT["transformation"]
    )

    dataloader_ = torch.utils.data.DataLoader(
        img_dataset_, 
        batch_size=50, shuffle=False
    )

    print("\n[Prediction] ...")
    # compute:
    y_pred = []
    imgs = []
    for X in dataloader_:
        with torch.no_grad():
            # to device:
            if device != None:
                X = X.to(device)
            # Predict:
            y_prediction = trained_net(X)
            # record:
            y_pred.extend(y_prediction.argmax(dim=1).cpu().detach().numpy())
        # torch.cuda.empty_cache()
        imgs = []
    print("     > DONE!")
            
    return y_pred

# %% Example main: ----- ----- ----- ----- ----- -----
MODEL_VER = "latest-v8-reduced-model"
MODEL_NAME = "best_state_dict_78:200.pth"
MODEL_ARCH = nn.Sequential(
    # Feature Extraction:
    # ResNet(BasicBlock, [0,1,1,1], num_classes=2), # ResNet reduced v8 - ResNet10 - ablation
    ResNet(BasicBlock, [1,1,1,1], num_classes=2), # ResNet reduced v8 - ResNet10
    # ResNet(BasicBlock, [1,2,3,2], num_classes=2), # ResNet reduced v7
    # ResNet(BasicBlock, [3,4,6,3], num_classes=2), # ResNet 34
    # Classifier:
    nn.Softmax(dim=1),
)
LIST_OF_TODO = [
#     {
#         "data_directory": "external_data",
#         "category": "COVID",
#         "model_directory": MODEL_VER,
#         "model_name": MODEL_NAME,
#     },
#     {
#         "data_directory": "external_data",
#         "category": "Lung_Opacity",
#         "model_directory": MODEL_VER,
#         "model_name": MODEL_NAME,
#     },
#     {
#         "data_directory": "external_data",
#         "category": "Normal",
#         "model_directory": MODEL_VER,
#         "model_name": MODEL_NAME,
#     },
#     {
#         "data_directory": "external_data",
#         "category": "Viral Pneumonia",
#         "model_directory": MODEL_VER,
#         "model_name": MODEL_NAME,
#     },
    {
        "data_directory": "data-latest",
        "category": "eval-custom-post",
        "model_directory": MODEL_VER,
        "model_name": MODEL_NAME,
    },
]

for TODO_ENTRY in LIST_OF_TODO:
    print("=== BEGIN ===")
    print(TODO_ENTRY)
    data_directory = TODO_ENTRY["data_directory"]
    category = TODO_ENTRY["category"]
    model_directory = TODO_ENTRY["model_directory"]
    model_name = TODO_ENTRY["model_name"]
    IMG_DIR = "/home/jx/JX_Project/covid-xray-detection/{}/{}".format(data_directory, category) 
    if data_directory == "data-latest":
        PROCESSING_NEEDED = False
        # do sth
        VALID_DATA_LUT = pd.read_csv(abs_data_path("pre-processed-[eval].txt"), sep=" ", header=None, names=LUT_HEADER)
        VALID_DATA_LUT["Y"] = class_to_binary(VALID_DATA_LUT["[class]"])
        imgs = filename_to_abspath(filenames=VALID_DATA_LUT["[filename]"], tag=category)
        report_status(data=VALID_DATA_LUT, tag="eval")
        y_ref = VALID_DATA_LUT["Y"]
    else:
        PROCESSING_NEEDED = True
        # path to original images:
        imgs = jx_lib.get_files(DIR=IMG_DIR)
        # % Validation:
        y_ref = np.zeros(len(imgs))
        if category == "COVID":
            y_ref = np.ones(len(imgs))
    # evaluate: 
    FILE_PATH = "/home/jx/JX_Project/covid-xray-detection/output/CUSTOM-MODEL/{}/models/{}".format(model_directory, model_name)
    output = eval(
        list_of_images=imgs, model_path=FILE_PATH, model_dict_type=("state_dict" in model_name), network_arch=MODEL_ARCH,
        PROCESSING_NEEDED=PROCESSING_NEEDED
    )
    OUT_FILE_PATH = "/home/jx/JX_Project/covid-xray-detection/output/CUSTOM-MODEL/{}/{}-{}.csv".format(model_directory, category, model_name)
    np.savetxt(OUT_FILE_PATH, output)
    #-report:
    print("accuracy: ",  (1- np.sum(np.abs(y_ref - output))/len(imgs))*100, " %")
    cm = confusion_matrix(output, y_ref)
    clr = classification_report(y_ref, output, target_names=LABEL_TO_INT_LUT)
    fig, status = jx_lib.make_confusion_matrix(cf=cm)
    model_output_path = "/home/jx/JX_Project/covid-xray-detection/output/CUSTOM-MODEL/{}".format(model_directory)
    fig.savefig("{}/confusion_matrix_[{}:{}].jpg".format(model_output_path, category, model_name), bbox_inches = 'tight')
    print("Best Classification Report:\n----------------------")
    print(clr)
    print("=== END === \n")

# %%
