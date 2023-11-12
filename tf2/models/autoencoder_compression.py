"""
Autoencoder Based Wireless Image Transmission model
"""
import tensorflow as tf
import tensorflow_compression as tfc
import numpy as np
from keras.layers import Input, PReLU
from keras.models import Model
from keras import layers


def build_model(c, snr):
    input_images = Input(shape=(32, 32, 3))
    # 1st convolutional layer
    conv1 = tfc.SignalConv2D(16, (5, 5), corr=True, strides_down=2, padding='same_zeros', use_bias=True)(input_images)
    prelu1 = PReLU(shared_axes=[1, 2])(conv1)
    # 2nd convolutional layer
    conv2 = tfc.SignalConv2D(32, (5, 5), corr=True, strides_down=2, padding='same_zeros', use_bias=True)(prelu1)
    prelu2 = PReLU(shared_axes=[1, 2])(conv2)
    # 3rd convolutional layer
    conv3 = tfc.SignalConv2D(32, (5, 5), corr=True, strides_down=1, padding='same_zeros', use_bias=True)(prelu2)
    prelu3 = PReLU(shared_axes=[1, 2])(conv3)
    # 4th convolutional layer
    conv4 = tfc.SignalConv2D(32, (5, 5), corr=True, strides_down=1, padding='same_zeros', use_bias=True)(prelu3)
    prelu4 = PReLU(shared_axes=[1, 2])(conv4)
    # 5th convolutional layer
    conv5 = tfc.SignalConv2D(c, (5, 5), corr=True, strides_down=1, padding='same_zeros', use_bias=True)(prelu4)
    encoder = PReLU(shared_axes=[1, 2])(conv5)

    real_prod = Channel(channel_type="awgn")(encoder, snr_db=snr)

    # 1st Deconvolutional layer
    decoder = tfc.SignalConv2D(32, (5, 5), corr=False, strides_up=1, padding='same_zeros', use_bias=True)(real_prod)
    decoder = PReLU(shared_axes=[1, 2])(decoder)
    # 2nd Deconvolutional layer
    decoder = tfc.SignalConv2D(32, (5, 5), corr=False, strides_up=1, padding='same_zeros', use_bias=True)(decoder)
    decoder = PReLU(shared_axes=[1, 2])(decoder)
    # 3rd Deconvolutional layer
    decoder = tfc.SignalConv2D(32, (5, 5), corr=False, strides_up=1, padding='same_zeros', use_bias=True)(decoder)
    decoder = PReLU(shared_axes=[1, 2])(decoder)
    # 4th Deconvolutional layer
    decoder = tfc.SignalConv2D(16, (5, 5), corr=False, strides_up=2, padding='same_zeros', use_bias=True)(decoder)
    decoder = PReLU(shared_axes=[1, 2])(decoder)
    # 5th Deconvolutional layer
    decoder = tfc.SignalConv2D(3, (5, 5), corr=False, strides_up=2, padding='same_zeros', use_bias=True, activation='sigmoid')(decoder)
    autoencoder = Model(input_images, decoder)
    return autoencoder


class Channel(layers.Layer):
    """"implement power constraint"""

    def __init__(self, channel_type, name="channel", **kwargs):
        super(Channel, self).__init__(name=name, **kwargs)
        self.channel_type = channel_type

    def get_config(self):
        config = super().get_config()
        config.update({
            "channel_type": self.channel_type,
        })
        return config

    def call(self, features, snr_db=None):
        inter_shape = tf.shape(features)
        f = layers.Flatten()(features)
        # convert to complex channel signal
        dim_z = tf.shape(f)[1] // 2
        z_in = tf.complex(f[:, :dim_z], f[:, dim_z:])
        # power constraint, the average complex symbol power is 1
        norm_factor = tf.reduce_sum(
            tf.math.real(z_in * tf.math.conj(z_in)), axis=1, keepdims=True
        )
        z_in_norm = z_in * tf.complex(
            tf.sqrt(tf.cast(dim_z, dtype=tf.float32) / norm_factor), 0.0
        )
        # Add channel noise
        if snr_db is None:
            raise Exception("This input snr should exist!")
        z_out = awgn(z_in_norm, snr_db)

        # convert signal back to intermediate shape
        z_out = tf.concat([tf.math.real(z_out), tf.math.imag(z_out)], 1)
        z_out = tf.reshape(z_out, inter_shape)
        return z_out


def awgn(x, snr_db):
    noise_stddev = tf.sqrt(10 ** (-snr_db / 10))
    noise_stddev = tf.complex(noise_stddev, 0.)
    noise = tf.complex(
        tf.random.normal(tf.shape(x), 0, 1 / np.sqrt(2)),
        tf.random.normal(tf.shape(x), 0, 1 / np.sqrt(2)),
    )
    return x + noise_stddev * noise
