
import timeit
import itertools
import skimage
import scipy
import pandas as pd
import numpy as np

import measures as ms
import sgessim


# my_image = io.imread("./images/img100.jpg")

# my_image[200:800, 200:800, :] = [255, 0, 0]

# plt.imshow(my_image)
# plt.show()


def iterate_images(original_image, image_array):
    
    image_list = image_array.files

    mse_list = []
    psnr_list = []
    ssim_list = []
    sg_essim_list = []

    mini = 1.0
    image_n = ""

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

        if sg_essim < mini:
            mini = sg_essim
            image_n = image_name
        
        print("\n")

    print(f"Min SG-ESSIM: {mini} for image: {image_n}\n")

    return mse_list, psnr_list, ssim_list, sg_essim_list





dataframe = pd.read_excel("./images/nits_iqa/Database/Score.xlsx", sheet_name="Sheet1")
dataframe = dataframe.sort_values(by=["Original Image Name", "Distorted Image Name"])
dataframe = dataframe.reset_index(drop=True)

mos_values = dataframe["Score"].values


original_images = []
NUMBER_OF_ORIGINAL_IMAGES = 9
mse_values = []
psnr_values = [] 
ssim_values = []
sg_essim_values = []


for i in range(1, NUMBER_OF_ORIGINAL_IMAGES + 1):
    original_image = skimage.io.imread(f"./images/nits_iqa/Database/I{i}.bmp")
    print(f"\nOriginal image no. {i} shape: {original_image.shape}\n")
    original_images.append(original_image)



time_start = timeit.default_timer()

for j in range(1, NUMBER_OF_ORIGINAL_IMAGES + 1):
    print(f"Collection no.: {j}")
    image_collection = skimage.io.imread_collection(f"./images/nits_iqa/Database/I{j}D*.bmp", conserve_memory=True)

    mse_v, psnr_v, ssim_v, sg_essim_v = iterate_images(original_images[j - 1], image_collection)
    mse_values.extend(mse_v)
    psnr_values.extend(psnr_v)
    ssim_values.extend(ssim_v)
    sg_essim_values.extend(sg_essim_v)

time_end = timeit.default_timer()
print(f"Time elapsed for processing: {time_end - time_start:.2f} seconds\n")

# Pearson’s linear correlation coefficient
pearson_coefficient, pe_p_value = scipy.stats.pearsonr(mos_values, sg_essim_values)

pearson_coefficient = np.round(pearson_coefficient, 3)

# Spearman’s rank-order correlation coefficient 
spearman_coefficient, sp_p_value = scipy.stats.spearmanr(mos_values, sg_essim_values)

spearman_coefficient = np.round(spearman_coefficient, 3)

# Kendall’s rank order correlation coefficient
kendall_coefficient, ke_p_value = scipy.stats.kendalltau(mos_values, sg_essim_values)

kendall_coefficient = np.round(kendall_coefficient, 3)

# Root mean square error
rmse = ms.rmse(mos_values, sg_essim_values)

rmse = np.round(rmse, 3)

print(f"\n===================\nPLCC: {pearson_coefficient}\n===================\n")
print(f"\n===================\nSROCC: {spearman_coefficient}\n===================\n")
print(f"\n===================\nKROCC: {kendall_coefficient}\n===================\n")
print(f"\n===================\nRMSE: {rmse}\n===================\n")



dataframe.insert(len(dataframe.columns), " ", " ", False)
dataframe.insert(len(dataframe.columns), "mse", "NaN", False)
dataframe.insert(len(dataframe.columns), "psnr", "NaN", False)
dataframe.insert(len(dataframe.columns), "ssim", "NaN", False)
dataframe.insert(len(dataframe.columns), "sg_essim", "NaN", False)

dataframe["mse"] = pd.Series(mse_values)
dataframe["psnr"] = pd.Series(psnr_values)
dataframe["ssim"] = pd.Series(ssim_values)
dataframe["sg_essim"] = pd.Series(sg_essim_values)


print(f"Dataframe after iteration:\n {dataframe.head(50)}\n\n")

print(f"MOS: {mos_values}\n")

dataframe.to_csv("./result_nits_iqa.csv", sep='\t', encoding='utf-8', index=False, header=True)
