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


# TODO: order and clean whatever this is
def ssim (original_image, deformed_image):
	MAX = np.iinfo(original_image.dtype).max

	c1 = (0.01 * MAX) ** 2
	c2 = (0.03 * MAX) ** 2

	numerator1 = 2 * np.mean(original_image) * np.mean(deformed_image) + c1
	denominator1 = (np.mean(original_image) ** 2) + (np.mean(deformed_image ** 2)) + c1
	numerator2 = 2 * (np.var(original_image) ** (1/2)) * (np.var(deformed_image) ** (1/2)) + c2
	denominator2 = np.var(original_image) + np.var(original_image) + c2

	return (numerator1 * numerator2) / (denominator1 * denominator2)





