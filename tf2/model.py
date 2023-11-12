# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 14:31:29 2019

@author: Danish
Wrapper File for 1. Compute pwr combined (Real, Imag), (Extract R & I parts) Generate single distribution, Separate Sending, (R&I)
"""

import os

from keras.callbacks import TensorBoard, ModelCheckpoint, Callback
from keras.optimizers import Adam

from metrics import psnr, ssim
from models.autoencoder_compression import build_model


class ModelCheckponitsHandler(Callback):
    def __init__(self, comp_ratio, snr_db, autoencoder, step):
        super(ModelCheckponitsHandler, self).__init__()
        self.comp_ratio = comp_ratio
        self.snr_db = snr_db
        self.step = step
        self.autoencoder = autoencoder

    def on_epoch_begin(self, epoch, logs=None):
        if epoch % self.step == 0:
            os.makedirs('./CKPT_ByEpochs/CompRatio_' +
                        str(self.comp_ratio)+'SNR'+str(self.snr_db), exist_ok=True)
            path = './CKPT_ByEpochs/CompRatio_' + \
                str(self.comp_ratio)+'SNR'+str(self.snr_db) + \
                '/Autoencoder_Epoch_'+str(epoch)+'.h5'
            self.autoencoder.save(path)
            print(f'\nModel Saved After {epoch} epochs.')


def TrainAutoEncoder(x_train, nb_epoch, comp_ratio, batch_size, c, snr, learning_rate, saver_step=50, verbose=2):
    ############################### Buliding Encoder ##############################
    ''' Correspondance of different arguments w.r.t to literature: filters = K, kernel_size = FxF, strides = S'''
    autoencoder = build_model(c=c, snr=snr)

    autoencoder.compile(optimizer=Adam(learning_rate=learning_rate),
                        loss='mse', metrics=['accuracy', psnr, ssim])
    autoencoder.summary()
    print('\t-----------------------------------------------------------------')
    print('\t|\t\t\t\t\t\t\t\t|')
    print('\t|\t\t\t\t\t\t\t\t|')
    print(f'\t| Training Parameters: Filter Size: {c}, Compression ratio: {comp_ratio} |')
    print(f'\t|\t\t\t  SNR: {snr} dB\t\t\t\t|')
    print('\t|\t\t\t\t\t\t\t\t|')
    print('\t|\t\t\t\t\t\t\t\t|')
    print('\t-----------------------------------------------------------------')
    tb = TensorBoard(
        log_dir=f'./Tensorboard/CompRatio{str(comp_ratio)}_SNR{str(snr)}')
    os.makedirs(
        f'./checkpoints/CompRatio{str(comp_ratio)}_SNR{str(snr)}', exist_ok=True)
    checkpoint = ModelCheckpoint(
        filepath=f'./checkpoints/CompRatio{str(comp_ratio)}_SNR{str(snr)}/Autoencoder.h5',
        monitor='val_loss', save_best_only=True)
    ckpt = ModelCheckponitsHandler(
        comp_ratio, snr, autoencoder, step=saver_step)
    history = autoencoder.fit(
        x=x_train, y=x_train, batch_size=batch_size, epochs=nb_epoch,
        callbacks=[tb, checkpoint, ckpt], validation_split=0.1, verbose=verbose)
    return history
