import numpy as np


def mse (array1, array2):
	return np.mean((np.subtract(array1, array2)) ** 2)


def rmse (array1, array2):
	return np.sqrt(mse(array1, array2))


def psnr (original_image, deformed_image):
	MAX = np.iinfo(original_image.dtype).max #maximum value of datarange calculated using image data type

	mse_value = mse(original_image, deformed_image)
	if mse_value == 0.:
		return np.inf
	return 10 * np.log10((MAX ** 2) / mse_value)





