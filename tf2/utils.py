import numpy as np

def normalize_pixels(train_data: np.ndarray, test_data: np.ndarray):
    # convert integer values to float
    train_norm = train_data.astype('float32')
    test_norm = test_data.astype('float32')
    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    return train_norm, test_norm


def Calculate_filters(comp_ratio, F=5, n=3072):
    """ Parameters
        ----------
        **comp_ratio**: Value of compression ratio i.e `k/n`

        **F**: Filter height/width both are same.

        **n** = Number of pixels in input image, calculated as `n = no_channels*img_height*img_width`

        Returns
        ----------
        **Number of filters required for the last Convolutional layer and first Transpose Convolutional layer for given compression ratio.**
        """
    channel_bandwidth = comp_ratio*n
    return int(channel_bandwidth/F**2)
