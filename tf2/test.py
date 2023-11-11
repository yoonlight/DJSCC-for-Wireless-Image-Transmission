from keras.datasets import cifar10

from utils import normalize_pixels
from metrics import ssim, psnr



(trainX, _), (testX, _) = cifar10.load_data()
x_train, x_test = normalize_pixels(trainX, testX)

def EvaluateModel(x_test, compression_ratios, snr, mode='multiple'):
    if mode == 'single':
        tf.keras.backend.clear_session()
        comp_ratio = compression_ratios
        path = './checkpoints/CompRatio{0}_SNR{1}/Autoencoder.h5'.format(
            comp_ratio, snr)
        autoencoder = load_model(
            path, custom_objects={'NormalizationNoise': NormalizationNoise})
        K.set_value(autoencoder.get_layer('normalization_noise_1').snr_db, snr)
        pred_images = autoencoder.predict(x_test)*255
        pred_images = pred_images.astype('uint8')
        ssim = ssim(testX, pred_images, multichannel=True)
        psnr = psnr(testX, pred_images)
        return pred_images, psnr, ssim
    elif mode == 'multiple':
        model_dic = {'SNR': [], 'Pred_Images': [], 'PSNR': [], 'SSIM': []}
        model_dic['SNR'].append(snr)
        for comp_ratio in compression_ratios:
            tf.keras.backend.clear_session()
            path = './checkpoints/CompRatio{0}_SNR{1}/Autoencoder.h5'.format(
            	comp_ratio, snr)
            autoencoder = load_model(
            	path, custom_objects={'NormalizationNoise': NormalizationNoise})
            K.set_value(autoencoder.get_layer(
            	'normalization_noise_1').snr_db, snr)
            pred_images = autoencoder.predict(x_test)*255
            pred_images = pred_images.astype('uint8')
            ssim = structural_similarity(testX, pred_images, multichannel=True)
            psnr = peak_signal_noise_ratio(testX, pred_images)
            model_dic['Pred_Images'].append(pred_images)
            model_dic['PSNR'].append(psnr)
            model_dic['SSIM'].append(ssim)
        return model_dic
