
import itertools
import skimage
import scipy
import numpy as np
import pandas as pd

import measures as ms
import sgessim
import ffs



def iterate_images(reference_image, image_array, console_log=False):
    
    image_list = image_array.files

    mse_list = []
    psnr_list = []
    ssim_list = []
    sg_essim_list = []
    ffs_list = []


    for image, image_name in itertools.zip_longest(image_array, image_list):
        
        mse = ms.mse(reference_image, image) # np.round(ms.mse(reference_image, image), 3)
        mse_list.append(mse)

        psnr = ms.psnr(reference_image, image) # np.round(ms.psnr(reference_image, image), 3)
        psnr_list.append(psnr)

        ssim = skimage.metrics.structural_similarity(reference_image, image, channel_axis=2) # np.round(skimage.metrics.structural_similarity(reference_image, image, channel_axis=2), 3)
        ssim_list.append(ssim)

        sg_essim = sgessim.sg_essim(reference_image, image) # np.round(sgessim.sg_essim(reference_image, image), 3)
        sg_essim_list.append(sg_essim)

        ffs_ = ffs.calculate_ffs(reference_image, image)
        ffs_list.append(ffs_)
        
        if console_log == True:
            print(f"Image: {image_name}")
            print(f"MSE: {mse}")
            print(f"PSNR: {psnr}")
            print(f"SSIM: {ssim}")
            print(f"SG-ESSIM: {sg_essim}")
            print(f"FFS: {ffs_}")
        
            print("\n")

    return mse_list, psnr_list, ssim_list, sg_essim_list, ffs_list


def get_coefficients(array1, array2):

    # Pearson’s linear correlation coefficient
    pearson_coefficient, pe_p_value = scipy.stats.pearsonr(array1, array2)

    pearson_coefficient = np.round(pearson_coefficient, 3)

    # Spearman’s rank-order correlation coefficient 
    spearman_coefficient, sp_p_value = scipy.stats.spearmanr(array1, array2)

    spearman_coefficient = np.round(spearman_coefficient, 3)

    # Kendall’s rank order correlation coefficient
    kendall_coefficient, ke_p_value = scipy.stats.kendalltau(array1, array2)

    kendall_coefficient = np.round(kendall_coefficient, 3)

    # Root mean square error
    rmse_error = ms.rmse(array1, array2)

    rmse_error = np.round(rmse_error, 3)

    return pearson_coefficient, spearman_coefficient, kendall_coefficient, rmse_error


def save_values_to_df(df, **kwargs):

    df.insert(len(df.columns), " ", " ", False)

    for key, value in kwargs.items():
        df.insert(len(df.columns), key, "NaN", False)
        df[key] = pd.Series(value)

    return df
