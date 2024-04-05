import os
import sys
import time
import warnings
from pathlib import Path
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

model_results = "nanopillar_denoise"
model_name = "n2v2_2d_srs"
model = N2V(config=None, name=model_name, basedir=model_results)
n_tiles = (2, 1)
input_dir = Path("images")
output_dir = Path("prediction")
for r, d, f in os.walk(input_dir):
    for file in f:
        filename = os.path.basename(file)
        img = imread(os.path.join(r, file))
        pred = model.predict(img, axes="YX", n_tiles=n_tiles)
        imsave(os.path.join(output_dir, filename), pred)
