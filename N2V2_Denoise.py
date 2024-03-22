import csv
import os
import shutil
import sys
import time
import warnings
import zipfile
from random import choice

import matplotlib.pyplot as plt
import numpy as np
import skimage
import tensorflow as tf
import wget
from csbdeep.utils import plot_history
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
from n2v.models import N2V, N2VConfig
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tifffile import imread, imsave

#! 1. Define the parameters for the training of the N2V model
use_pretrained_model = False
pretrained_model_path = ""
# obtain location of the pretrained model:
if use_pretrained_model:
    model_file_path = pretrained_model_path
    if not os.path.exists(model_file_path):
        print("Model does not exist.")
        use_pretrained_model = False
    else:
        pass

if use_pretrained_model:
    print("Model", os.path.basename(pretrained_model_path), "was successfully loaded")
else:
    print("A pretrained model will not be used.")

#! 2. Define the source of the training data
training_source = "images"
model_results = "model_results"
model_name = "N2V2"
# call random image:
random_img = choice(os.listdir(training_source))
# check file type
filename, filetype = os.path.splitext(random_img)
if filetype == ".tiff":
    filext = "*.tiff"
elif filetype == ".tif":
    filext = "*.tif"
else:
    print("The images you are trying to use are the wrong file type. File type of your images:", filetype)
    sys.exit()
# if pass, extract data
datagen = N2V_DataGenerator()
imgs = datagen.load_imgs_from_directory(directory=training_source, dims="ZYX")
print("Images loaded successfully")

#! 3. Define the parameters for the training of the N2V model
number_of_epochs = 50
patch_size = 64
patch_height = 4
# check patch size:
x = imread(training_source + "/" + random_img)
img_z = x.shape[0]
mid_plane = int(img_z / 2) + 1
img_y = x.shape[1]
img_x = x.shape[2]
if patch_size > min(img_y, img_x):
    patch_size = min(img_y, img_x)
    print("Patch size was larger than the dimensions of the training images; patch size modified.")
if not patch_size % 8 == 0:
    patch_size = (int(patch_size / 8) - 1) * 8
    print("Patch size not divisible by 8; patch size modified.")
if not patch_size >= 64:
    patch_size = 64
    print("Patch size was smaller than 64; patch size modified.")
if patch_height > img_z:
    patch_height = img_z
    print("Patch height was larger than the dimensions of the training images; patch height modified.")
if not patch_height % 4 == 0:
    patch_height = (int(patch_height / 4) - 1) * 4
    if patch_height == 0:
        patch_height = 4
    print("Patch height not divisible by 4; patch height modified.")

batch_size = 128
number_of_steps = 100
initial_learning_rate = 0.0004
percent_validation = 10
data_augmentation = True
print("Parameters set.")

#! 4. Prepare the model for training
patch_dims = (patch_height, patch_size, patch_size)
patches = datagen.generate_patches_from_list(imgs, shape=patch_dims, augment=data_augmentation)
