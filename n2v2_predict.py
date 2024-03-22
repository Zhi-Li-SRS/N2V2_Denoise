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

model_name = "n2v2_3D_flr"
model_results = "model_results"
model = N2V(config=None, name=model_name, basedir=model_results)
model.load_weights(os.path.join(model_results, "n2v2_3D_flr", "weights_last.h5"))
img_path = "images/3_ku80_400mw_zoom4.tif"
img = imread(img_path)
pred = model.predict(img, axes="ZYX", n_tiles=(2, 4, 4))
imsave("prediction/3_ku80_400mw_zoom4.tif", pred)
