
import timeit
import skimage
import pandas as pd

import utils


NUMBER_OF_REFERENCE_IMAGES = 9

reference_images = []
mse_values = []
psnr_values = [] 
ssim_values = []
sg_essim_values = []
ffs_values = []


# my_image = io.imread("./images/img100.jpg")

# my_image[200:800, 200:800, :] = [255, 0, 0]

# plt.imshow(my_image)
# plt.show()


dataframe = pd.read_excel("./images/nits_iqa/Database/Score.xlsx", sheet_name="Sheet1")
dataframe = dataframe.sort_values(by=["Original Image Name", "Distorted Image Name"])
dataframe = dataframe.reset_index(drop=True)

mos_values = dataframe["Score"].values



for i in range(1, NUMBER_OF_REFERENCE_IMAGES + 1):
    ref_image = skimage.io.imread(f"./images/nits_iqa/Database/I{i}.bmp")
    print(f"\nOriginal image no. {i} shape: {ref_image.shape}\n")
    reference_images.append(ref_image)


time_start = timeit.default_timer()

for j in range(1, NUMBER_OF_REFERENCE_IMAGES + 1):
    print(f"Collection no.: {j}")
    image_collection = skimage.io.imread_collection(f"./images/nits_iqa/Database/I{j}D*.bmp", conserve_memory=True)

    mse_v, psnr_v, ssim_v, sg_essim_v, ffs_v = utils.iterate_images(reference_images[j - 1], image_collection, console_log=True)
    mse_values.extend(mse_v)
    psnr_values.extend(psnr_v)
    ssim_values.extend(ssim_v)
    sg_essim_values.extend(sg_essim_v)
    ffs_values.extend(ffs_v)

time_end = timeit.default_timer()
print(f"Time elapsed for processing: {time_end - time_start:.2f} seconds\n")


plcc, srocc, krocc, rmse = utils.get_coefficients(mos_values, sg_essim_values)

print(f"=======================================================\n")
print(f"=======================SG-ESSIM========================\n")
print(f"=======================================================\n")

print(f"\n===================\nPLCC: {plcc}\n===================\n")
print(f"\n===================\nSROCC: {srocc}\n===================\n")
print(f"\n===================\nKROCC: {krocc}\n===================\n")
print(f"\n===================\nRMSE: {rmse}\n===================\n")

plcc, srocc, krocc, rmse = utils.get_coefficients(mos_values, ffs_values)

print(f"=======================================================\n")
print(f"=========================FFS===========================\n")
print(f"=======================================================\n")

print(f"\n===================\nPLCC: {plcc}\n===================\n")
print(f"\n===================\nSROCC: {srocc}\n===================\n")
print(f"\n===================\nKROCC: {krocc}\n===================\n")
print(f"\n===================\nRMSE: {rmse}\n===================\n")

plcc, srocc, krocc, rmse = utils.get_coefficients(mos_values, ssim_values)

print(f"=======================================================\n")
print(f"=======================SSIM========================\n")
print(f"=======================================================\n")

print(f"\n===================\nPLCC: {plcc}\n===================\n")
print(f"\n===================\nSROCC: {srocc}\n===================\n")
print(f"\n===================\nKROCC: {krocc}\n===================\n")
print(f"\n===================\nRMSE: {rmse}\n===================\n")


value_dictionary = {
    "mse": mse_values,
    "psnr": psnr_values,
    "ssim": ssim_values,
    "sg_essim": sg_essim_values,
    "ffs": ffs_values
}

new_df = utils.save_values_to_df(dataframe, **value_dictionary)

print(f"Dataframe after iteration:\n {new_df.head(50)}\n\n")

new_df.to_csv("./result_nits_iqa.csv", sep='\t', encoding='utf-8', index=False, header=True)
