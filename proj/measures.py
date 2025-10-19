import numpy as np
from math import log2, log10


def mse (original_image, deformed_image):
	return np.mean((original_image.astype(np.float64) - deformed_image.astype(np.float64)) ** 2)


def rmse (original_image, deformed_image):
	return np.sqrt(mse(original_image, deformed_image))


def psnr (original_image, deformed_image):
	MAX = np.iinfo(original_image.dtype).max #maximum value of datarange calculated using image data type

	mse_value = mse(original_image, deformed_image)
	if mse_value == 0.:
		return np.inf
	return 10 * np.log10((MAX ** 2) / mse_value)


# def ssim (original_image, deformed_image):

