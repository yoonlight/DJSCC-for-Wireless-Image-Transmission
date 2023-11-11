# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 00:39:40 2019

@author: Danish
1. Compute pwr combined (Real, Imag), (Extract R & I parts) Generate single distribution, Separate Sending, (R&I)
"""

from keras.datasets import cifar10
import tensorflow as tf

from model import TrainAutoEncoder
from utils import normalize_pixels, Calculate_filters

(trainX, _), (testX, _) = cifar10.load_data()


# normalizing the training and test data
x_train, x_test = normalize_pixels(trainX, testX)
# compression_ratios = [0.06, 0.09, 0.17, 0.26, 0.34, 0.43, 0.49]
compression_ratios = [1/12, 1/6]
SNR = [0, 10, 20]
for comp_ratio in compression_ratios:
    tf.keras.backend.clear_session()
    c = Calculate_filters(comp_ratio)
    print(f'---> System Will Train, Compression Ratio: {str(comp_ratio)}. <---')
    _ = TrainAutoEncoder(
        x_train, nb_epoch=200, comp_ratio=comp_ratio,
        batch_size=64, c=c, snr=10, learning_rate=1e-3, saver_step=50)
