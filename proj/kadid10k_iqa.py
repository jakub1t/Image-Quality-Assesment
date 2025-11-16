
import timeit
import skimage
import scipy
import pandas as pd
import numpy as np

import measures as ms
import sgessim
import utils


NUMBER_OF_ORIGINAL_IMAGES = 81

original_images = []
mse_values = []
psnr_values = [] 
ssim_values = []
sg_essim_values = []



dataframe = pd.read_csv("./images/kadid10k/dmos.csv", sep=",")

dmos_values = dataframe["dmos"].values


for i in range(1, NUMBER_OF_ORIGINAL_IMAGES + 1):
    if i < 10:
        original_image = skimage.io.imread(f"./images/kadid10k/images/I0{i}.png")
    else:
        original_image = skimage.io.imread(f"./images/kadid10k/images/I{i}.png")
    original_images.append(original_image)



time_start = timeit.default_timer()

for j in range(1, NUMBER_OF_ORIGINAL_IMAGES + 1):
    print(f"Collection no.: {j}")
    if j < 10:
        image_collection = skimage.io.imread_collection(f"./images/kadid10k/images/I0{j}_*.png", conserve_memory=True)
    else:
        image_collection = skimage.io.imread_collection(f"./images/kadid10k/images/I{j}_*.png", conserve_memory=True)

    mse_v, psnr_v, ssim_v, sg_essim_v = utils.iterate_images(original_images[j - 1], image_collection)
    mse_values.extend(mse_v)
    psnr_values.extend(psnr_v)
    ssim_values.extend(ssim_v)
    sg_essim_values.extend(sg_essim_v)

time_end = timeit.default_timer()
print(f"Time elapsed for processing: {time_end - time_start:.2f} seconds\n")



plcc, srocc, krocc, rmse = utils.get_coefficients(dmos_values, sg_essim_values)

print(f"\n===================\nPLCC: {plcc}\n===================\n")
print(f"\n===================\nSROCC: {srocc}\n===================\n")
print(f"\n===================\nKROCC: {krocc}\n===================\n")
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

dataframe.to_csv("./result_kadid10k_iqa.csv", sep='\t', encoding='utf-8', index=False, header=True)
