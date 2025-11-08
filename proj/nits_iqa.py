
import itertools
import skimage
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import other_measures as ms
import sgessim

"""

Image data types:

uint8 ->  0 to 255
uint16 -> 0 to 65535
uint32 -> 0 to (2^32 - 1)

int8 -> -128 to 127
int16 -> -32768 to 32767
int32 -> -2^31 to (2^31 - 1)

float -> -1 to 1 or 0 to 1

Functions that convert images to desired data type and properly rescale their values
img_as_float - convert to 64-bit float
img_as_ubyte - convert to 8-bit uint
img_as_uint - convert to 16-bit uint
img_as_int - convert to 16-bit int

"""


# my_image = io.imread("./images/img100.jpg")

# my_image[200:800, 200:800, :] = [255, 0, 0]

# plt.imshow(my_image)
# plt.show()


def iterate_images(original_image, image_array):
    
    image_list = image_array.files

    mse_list = []
    rmse_list = []
    psnr_list = []
    sg_essim_list = []

    mini = 1.0
    image_n = ""

    for image, image_name in itertools.zip_longest(image_array, image_list):
        print(f"Image: {image_name}")
        
        mse = np.round(ms.mse(original_image, image), 3)
        mse_list.append(mse)
        print(f"MSE: {mse}")

        rmse = np.round(ms.rmse(original_image, image), 3)
        rmse_list.append(rmse)
        # print(f"RMSE: {rmse}")

        psnr = np.round(ms.psnr(original_image, image), 3)
        psnr_list.append(psnr)
        # print(f"PSNR: {psnr}")

        # print(f"SSIM: {ms.ssim(original_image, image)}")
        # print(f"Skimage SSIM: {skimage.metrics.structural_similarity(original_image, image, channel_axis=2)}")

        sg_essim = np.round(sgessim.sg_essim(original_image, image), 3)
        sg_essim_list.append(sg_essim)
        # print(f"SG-ESSIM: {sg_essim}")

        if sg_essim < mini:
            mini = sg_essim
            image_n = image_name

    print(f"Min SG-ESSIM: {mini} for image: {image_n}")

    return mse_list, rmse_list, psnr_list, sg_essim_list


dataframe = pd.read_excel("./images/nits_iqa/Database/Score.xlsx", sheet_name="Sheet1")
dataframe = dataframe.sort_values(by=["Original Image Name", "Distorted Image Name"])
dataframe = dataframe.reset_index(drop=True)


# all_deformed_images = []
original_images = []
NUMBER_OF_ORIGINAL_IMAGES = 9
mse_values = []
rmse_values = [] 
psnr_values = [] 
sg_essim_values = []


for i in range(1, NUMBER_OF_ORIGINAL_IMAGES + 1):
    original_image = skimage.io.imread(f"./images/nits_iqa/Database/I{i}.bmp")
    print(f"\nOriginal image no. {i} shape: {original_image.shape}\n")
    original_images.append(original_image)

for j in range(1, NUMBER_OF_ORIGINAL_IMAGES + 1):
    print(f"Collection no.: {j}")
    image_collection = skimage.io.imread_collection(f"./images/nits_iqa/Database/I{j}D*.bmp", conserve_memory=True)
    # all_deformed_images.append(image_collection)

    mse_v, rmse_v, psnr_v, sg_essim_v = iterate_images(original_images[j - 1], image_collection)
    mse_values.extend(mse_v)
    rmse_values.extend(rmse_v)
    psnr_values.extend(psnr_v)
    sg_essim_values.extend(sg_essim_v)



# dataframe.style.set_properties(**{"background-color": "red"}, subset=["mse", "rmse", "psnr", "sg_essim"])

dataframe.insert(len(dataframe.columns), " ", " ", False)
dataframe.insert(len(dataframe.columns), "mse", "NaN", False)
dataframe.insert(len(dataframe.columns), "rmse", "NaN", False)
dataframe.insert(len(dataframe.columns), "psnr", "NaN", False)
dataframe.insert(len(dataframe.columns), "sg_essim", "NaN", False)

dataframe["mse"] = pd.Series(mse_values)
dataframe["rmse"] = pd.Series(rmse_values)
dataframe["psnr"] = pd.Series(psnr_values)
dataframe["sg_essim"] = pd.Series(sg_essim_values)


print(f"Dataframe after iteration:\n {dataframe.head(50)}")

dataframe.to_csv("./result.csv", sep='\t', encoding='utf-8', index=False, header=True)
