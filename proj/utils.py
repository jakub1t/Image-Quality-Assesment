
import itertools
import skimage
import scipy
import numpy as np

import measures as ms
import sgessim



def iterate_images(original_image, image_array):
    
    image_list = image_array.files

    mse_list = []
    psnr_list = []
    ssim_list = []
    sg_essim_list = []


    for image, image_name in itertools.zip_longest(image_array, image_list):
        print(f"Image: {image_name}")
        
        mse = np.round(ms.mse(original_image, image), 3)
        mse_list.append(mse)
        # print(f"MSE: {mse}")

        psnr = np.round(ms.psnr(original_image, image), 3)
        psnr_list.append(psnr)
        # print(f"PSNR: {psnr}")

        ssim = np.round(skimage.metrics.structural_similarity(original_image, image, channel_axis=2), 3)
        ssim_list.append(ssim)
        print(f"SSIM: {ssim}")

        sg_essim = np.round(sgessim.sg_essim(original_image, image), 3)
        sg_essim_list.append(sg_essim)
        print(f"SG-ESSIM: {sg_essim}")
        
        print("\n")

    return mse_list, psnr_list, ssim_list, sg_essim_list


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