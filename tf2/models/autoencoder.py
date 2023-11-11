"""
Autoencoder Based Wireless Image Transmission model
"""
from keras.layers import Conv2D, Input, Conv2DTranspose, PReLU
# from keras.layers import UpSampling2D, Cropping2D
from keras.models import Model

from models.channel import NormalizationNoise


def build_model(c, snr):
    input_images = Input(shape=(32, 32, 3))
    # 1st convolutional layer
    conv1 = Conv2D(filters=16, kernel_size=(5, 5), strides=2,
                   padding='same', kernel_initializer='he_normal')(input_images)
    prelu1 = PReLU()(conv1)
    # 2nd convolutional layer
    conv2 = Conv2D(filters=32, kernel_size=(5, 5), strides=2,
                   padding='same', kernel_initializer='he_normal')(prelu1)
    prelu2 = PReLU()(conv2)
    # 3rd convolutional layer
    conv3 = Conv2D(filters=32, kernel_size=(5, 5), strides=1,
                   padding='same', kernel_initializer='he_normal')(prelu2)
    prelu3 = PReLU()(conv3)
    # 4th convolutional layer
    conv4 = Conv2D(filters=32, kernel_size=(5, 5), strides=1,
                   padding='same', kernel_initializer='he_normal')(prelu3)
    prelu4 = PReLU()(conv4)
    # 5th convolutional layer
    conv5 = Conv2D(filters=c, kernel_size=(5, 5), strides=1,
                   padding='same', kernel_initializer='he_normal')(prelu4)
    encoder = PReLU()(conv5)

    real_prod = NormalizationNoise(snr_db=snr)(encoder)

    ############################### Building Decoder ##############################
    # 1st Deconvolutional layer
    decoder = Conv2DTranspose(filters=32, kernel_size=(
        5, 5), strides=1, padding='same', kernel_initializer='he_normal')(real_prod)
    decoder = PReLU()(decoder)
    # 2nd Deconvolutional layer
    decoder = Conv2DTranspose(filters=32, kernel_size=(
        5, 5), strides=1, padding='same', kernel_initializer='he_normal')(decoder)
    decoder = PReLU()(decoder)
    # 3rd Deconvolutional layer
    decoder = Conv2DTranspose(filters=32, kernel_size=(
        5, 5), strides=1, padding='same', kernel_initializer='he_normal')(decoder)
    decoder = PReLU()(decoder)
    # 4th Deconvolutional layer
    decoder = Conv2DTranspose(filters=16, kernel_size=(
        5, 5), strides=2, padding='same', kernel_initializer='he_normal')(decoder)
    decoder = PReLU()(decoder)
    # decoder_up = UpSampling2D((2,2))(decoder)
    # 5th Deconvolutional layer
    decoder = Conv2DTranspose(
        filters=3, kernel_size=(5, 5), strides=2, padding='same',
        kernel_initializer='he_normal', activation='sigmoid')(decoder)
    # decoder_up = UpSampling2D((2, 2))(decoder)
    # decoder = Cropping2D(cropping=((13, 13), (13, 13)))(decoder_up)
    ############################### Buliding Models ###############################
    autoencoder = Model(input_images, decoder)
    return autoencoder
