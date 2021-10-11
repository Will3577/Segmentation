

from Code.utils.metricfunctions import dice_coef,f1
from Code.utils.lossfunctions import *

import os
import tensorflow as tf
from torch.utils.data import DataLoader, random_split
import segmentation_models_pytorch as smp

from skimage.util.shape import view_as_windows
import json
import cv2
import torch
import torch.nn as nn
from Code.utils.data_loading import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

with open('./config.json') as config_file:
    config = json.load(config_file)
# print (config)
im_width = config['im_width']
im_height = config['im_height']
patch_width = config['patch_width']
patch_height = config['patch_height']
Epochs = config['Epochs']
batch_size = config['Batch']

TRAIN_PATH_IMAGES = config['TRAIN_PATH_IMAGES']
TRAIN_PATH_GT = config['TRAIN_PATH_GT']
TEST_PATH_IMAGES = config['TEST_PATH_IMAGES']
TEST_PATH_GT = config['TEST_PATH_GT']


# 1. Create dataset
img_scale = [im_width,im_height]
dataset = BasicDataset(TRAIN_PATH_IMAGES, TRAIN_PATH_GT, img_scale)

# 2. Split into train / validation partitions
n_val = 14 #int(len(dataset) * val_percent)
n_train = len(dataset) - n_val
train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

# 3. Create data loaders
loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
train_loader = DataLoader(train_set, shuffle=True, **loader_args)
val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)



model = smp.Unet(
        encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1,                      # model output channels (number of classes in your dataset)
    )







