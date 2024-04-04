import os

import tensorflow as tf
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
from n2v.models import N2V
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tifffile import imread, imsave

model_results = "model_results"
model_name = "n2v2_3D_flr"
model = N2V(config=None, name=model_name, basedir=model_results)
n_tiles = (2, 4, 4)
input_dir = "input"
output_dir = "prediction"
for r, d, f in os.walk(input_dir):
    for file in f:
        filename = os.path.basename(file)
        img = imread(os.path.join(r, file))
        pred = model.predict(img, axes="ZYX", n_tiles=n_tiles)
        imsave(os.path.join(output_dir, filename), pred)
