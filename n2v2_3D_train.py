import os
import sys
import time
import warnings
from random import choice

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

#! 1. Define the source of the training data
if not sys.warnoptions:
    warnings.filterwarnings("ignore")

training_source = "train_images"
model_results = "model_results"
model_name = "n2v2_3D_srs"
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

#! 2. Define the parameters for the training of the N2V model
number_of_epochs = 150
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
initial_learning_rate = 0.0004
percent_validation = 10
data_augmentation = False
print("Parameters set.")

#! 3. Using weights from pre-trained model as initial weights
use_pretrained_model = False
pretrained_model_choice = "n2v2_3D_flr"
weights_choice = "last"
pretrained_model_path = "model_results"
if use_pretrained_model:
    if pretrained_model_choice == pretrained_model_choice:
        h5_file_path = os.path.join(pretrained_model_path, "weights_" + weights_choice + ".weights.h5")
    if not os.path.exists(h5_file_path):
        print("WARNING: weights_last.h5 pretrained model does not exist")
        use_pretrained_model = False


#! 4. Prepare the model for training
patch_dims = (patch_height, patch_size, patch_size)
datagen = N2V_DataGenerator()
patches = datagen.generate_patches_from_list(imgs, shape=patch_dims)
threshold = int(len(patches) * (percent_validation / 100))
X = patches[threshold:]
X_val = patches[:threshold]

config = N2VConfig(
    X,
    unet_kern_size=3,
    train_steps_per_epoch=int(X.shape[0] / batch_size) + 1,
    train_epochs=number_of_epochs,
    train_loss="mse",
    batch_norm=True,
    train_batch_size=batch_size,
    n2v_perc_pix=0.198,
    n2v_patch_shape=patch_dims,
    n2v_manipulator="uniform_withCP",
    train_learning_rate=initial_learning_rate,
    single_net_per_channel=False,
    n2v_neighborhood_radius=5,
)

vars(config)

model = N2V(config=config, name=model_name, basedir=model_results)
if use_pretrained_model:
    model.load_weights(h5_file_path)
    print("Pretrained model loaded.")
print("Configuration complete. Ready to train.")

#! 5. Train the model
warnings.filterwarnings("ignore")
start = time.time()
history = model.train(X, X_val)

print("Training Complete")
print("Time Elapsed:", time.strftime("%H:%M:%S", time.gmtime(time.time() - start)))
print("Model was successfully exported in folder:", model_results)
