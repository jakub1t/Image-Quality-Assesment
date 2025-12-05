
import numpy as np
from pandas import Series
from itertools import zip_longest, repeat
from skimage.metrics import structural_similarity
from scipy.stats import pearsonr, spearmanr, kendalltau

from concurrent.futures import ProcessPoolExecutor

from sgessim import sg_essim
from ffs import calculate_ffs


def mse (array1, array2):
	return np.mean((np.subtract(array1, array2)) ** 2)


def psnr (reference_image, deformed_image):
	MAX = np.iinfo(reference_image.dtype).max #maximum value of datarange calculated using image data type

	mse_value = mse(reference_image, deformed_image)
	if mse_value == 0.:
		return np.inf
	return 10 * np.log10((MAX ** 2) / mse_value)


def calculate_quality_from_measures(reference_image, image):
        
    mse_val = mse(reference_image, image)

    psnr_val = psnr(reference_image, image)

    ssim_val = structural_similarity(reference_image, image, channel_axis=2)

    sg_essim_val = sg_essim(reference_image, image)

    ffs_val = calculate_ffs(reference_image, image)
        

    return mse_val, psnr_val, ssim_val, sg_essim_val, ffs_val


def iterate_images(reference_image, image_array, console_log=False):
    
    image_list = image_array.files
    
    mse_list = []
    psnr_list = []
    ssim_list = []
    sg_essim_list = []
    ffs_list = []

    with ProcessPoolExecutor() as executor:
        for i, result in enumerate(executor.map(calculate_quality_from_measures, repeat(reference_image), image_array)):
            mse_val, psnr_val, ssim_val, sg_essim_val, ffs_val = result
            mse_list.append(mse_val)
            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)
            sg_essim_list.append(sg_essim_val)
            ffs_list.append(ffs_val)

            if console_log == True:
                print(result)
                print(f"Image: {image_list[i]}")
                print(f"MSE: {mse_val}")
                print(f"PSNR: {psnr_val}")
                print(f"SSIM: {ssim_val}")
                print(f"SG-ESSIM: {sg_essim_val}")
                print(f"FFS: {ffs_val}")
            
                print("\n")

    return mse_list, psnr_list, ssim_list, sg_essim_list, ffs_list


def get_coefficients(array1, array2):

    # Pearson’s linear correlation coefficient
    pearson_coefficient, _ = pearsonr(array1, array2)

    pearson_coefficient = np.round(pearson_coefficient, 3)

    # Spearman’s rank-order correlation coefficient 
    spearman_coefficient, _ = spearmanr(array1, array2)

    spearman_coefficient = np.round(spearman_coefficient, 3)

    # Kendall’s rank order correlation coefficient
    kendall_coefficient, _ = kendalltau(array1, array2)

    kendall_coefficient = np.round(kendall_coefficient, 3)

    return pearson_coefficient, spearman_coefficient, kendall_coefficient


def save_values_to_df(df, **kwargs):

    df.insert(len(df.columns), " ", " ", False)

    for key, value in kwargs.items():
        df.insert(len(df.columns), key, "NaN", False)
        df[key] = Series(value)

    return df
