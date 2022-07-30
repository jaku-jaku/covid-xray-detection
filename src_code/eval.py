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
from typing import List
import sys, os
local_module_path = os.path.dirname(os.path.abspath(__file__))
if local_module_path not in sys.path:
    sys.path.append(local_module_path)
from PIL import Image

import matplotlib.pyplot as plt

# Classical CV:
import cv2

# CNN modules:
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset

import jx_lib
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
    

def img_batch_conversion(PATH_LUT, OUT_DIR, RANDOM_AUGMENTATION=False, size=None):
    jx_lib.create_folder(DIR=OUT_DIR)
    new_path_list=[]
    counter = 0
    for img_path, file_name in zip(PATH_LUT["img_abs_path"], PATH_LUT["[filename]"]):
        counter += 1
        print("\r   >[{}/{}]".format(counter,len(PATH_LUT["img_abs_path"])),  end='')
        out_path = "{}/{}".format(OUT_DIR, file_name)
        img = cv2.imread(img_path)
            
        # basic morphological operator
        kernel = np.ones((5, 5), 'uint8')
        img1 = cv2.dilate(img, kernel, iterations=5)
        img2 = cv2.erode(img, kernel, iterations=5)
        
        img_new = np.dstack((img[:,:,0], img1[:,:,1], img2[:,:,2]))
        
        if size is not None:
            # fit:
            w, h, c= img_new.shape
            if w <= h:
                nw = size[0]
                nh = int(size[0]/w * h)
            else:
                nh = size[0]
                nw = int(size[0] / h * w)
            img_new  = cv2.resize(img_new, (nw, nh))
            # center crop:
            w, h, c= img_new.shape
            a = int((w-size[0])/2)
            b = int((h-size[1])/2)
            img_new = img_new[a:a+size[0], b:b+size[1]]
            
        # print(out_path)
        cv2.imwrite(out_path, img_new)
        new_path_list.append(out_path)
        
    return new_path_list
# %% Evaluation Script
def eval(
    list_of_images  :List[str], 
    model_path      :str, 
    size            :tuple = (320,320),
    model_dict_type :bool = False, # if saved as model dictionary
    folder_name_for_pre_process_cache: str = "_reduced",
) -> List[int]:
    # let's pre-process images first:
    print("[Pre-processing] image conversion:")
    PATH_LUT = {}
    PATH_LUT["img_abs_path"]=list_of_images
    PATH_LUT["[filename]"]=[ os.path.basename(path) for path in list_of_images ]
    OUT_DIR = os.path.dirname(list_of_images[0]) + folder_name_for_pre_process_cache
    LIST_REDUCED_IMG_PATH = img_batch_conversion(PATH_LUT=PATH_LUT, OUT_DIR=OUT_DIR, size=(320,320))

    # load models:
    if model_dict_type:
        trained_net = CUSTOM_MODEL_DICT["model"]
        state_dict = torch.load(model_path)
        trained_net.load_state_dict(state_dict)
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
# path to original images:
imgs = ["/home/jx/JX_Project/covid-xray-detection/data/competition_test/{}.png".format(id) for id in range(1, 401)]
# evaluate:
output = eval(
    list_of_images=imgs, 
    model_path="/home/jx/JX_Project/covid-xray-detection/output/CUSTOM-MODEL/v6-custom-with-aug-10/models/best_model_138.pth",
)
print(output)

# % Validation:
y_ref = np.loadtxt("/home/jx/JX_Project/covid-xray-detection/output/CUSTOM-MODEL/v6-custom-with-aug-10/y_pred[best[107_200]].txt")
print("diff: ", np.sum(np.abs(y_ref - output)))

