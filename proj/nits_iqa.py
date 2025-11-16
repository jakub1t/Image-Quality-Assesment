
import timeit
import skimage
import scipy
import pandas as pd
import numpy as np

import measures as ms
import sgessim
import utils


NUMBER_OF_ORIGINAL_IMAGES = 9

original_images = []
mse_values = []
psnr_values = [] 
ssim_values = []
sg_essim_values = []


# my_image = io.imread("./images/img100.jpg")

# my_image[200:800, 200:800, :] = [255, 0, 0]

# plt.imshow(my_image)
# plt.show()


dataframe = pd.read_excel("./images/nits_iqa/Database/Score.xlsx", sheet_name="Sheet1")
dataframe = dataframe.sort_values(by=["Original Image Name", "Distorted Image Name"])
dataframe = dataframe.reset_index(drop=True)

mos_values = dataframe["Score"].values



for i in range(1, NUMBER_OF_ORIGINAL_IMAGES + 1):
    original_image = skimage.io.imread(f"./images/nits_iqa/Database/I{i}.bmp")
    print(f"\nOriginal image no. {i} shape: {original_image.shape}\n")
    original_images.append(original_image)


time_start = timeit.default_timer()

for j in range(1, NUMBER_OF_ORIGINAL_IMAGES + 1):
    print(f"Collection no.: {j}")
    image_collection = skimage.io.imread_collection(f"./images/nits_iqa/Database/I{j}D*.bmp", conserve_memory=True)

    mse_v, psnr_v, ssim_v, sg_essim_v = utils.iterate_images(original_images[j - 1], image_collection)
    mse_values.extend(mse_v)
    psnr_values.extend(psnr_v)
    ssim_values.extend(ssim_v)
    sg_essim_values.extend(sg_essim_v)

time_end = timeit.default_timer()
print(f"Time elapsed for processing: {time_end - time_start:.2f} seconds\n")


plcc, srocc, krocc, rmse = utils.get_coefficients(mos_values, sg_essim_values)

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

dataframe.to_csv("./result_nits_iqa.csv", sep='\t', encoding='utf-8', index=False, header=True)
