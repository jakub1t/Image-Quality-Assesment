import numpy as np
from math import log2, log10
import scipy



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
def ssim (original_image, deformed_image, k1=0.01, k2=0.03, k3=0.05):
	MAX = np.iinfo(original_image.dtype).max

	c1 = (k1 * MAX) ** 2
	c2 = (k2 * MAX) ** 2
	c3 = (k3 * MAX) ** 2

	luminance_numerator = 2 * np.mean(original_image) * np.mean(deformed_image) + c1
	luminance_denominator = (np.mean(original_image) ** 2) + (np.mean(deformed_image) ** 2) + c1
	contrast_numerator = 2 * np.std(original_image) * np.std(deformed_image) + c2
	contrast_denominator = np.var(original_image) + np.var(deformed_image) + c2
	structure_numerator = covariance(original_image, deformed_image) + c3
	structure_denominator = np.std(original_image) * np.std(deformed_image) + c3

	return (luminance_numerator * contrast_numerator * structure_numerator) / (luminance_denominator * contrast_denominator * structure_denominator)


def covariance (array1, array2):
	array1_mean, array2_mean = np.mean(array1), np.mean(array2)
	return np.sum(((array1 - array1_mean) * (array2 - array2_mean)) / (len(array1 - 1)))



