
import timeit
import skimage
import pandas as pd

import utils


NUMBER_OF_REFERENCE_IMAGES = 81

reference_images = []
mse_values = []
psnr_values = [] 
ssim_values = []
sg_essim_values = []



dataframe = pd.read_csv("./images/kadid10k/dmos.csv", sep=",")

dmos_values = dataframe["dmos"].values


for i in range(1, NUMBER_OF_REFERENCE_IMAGES + 1):
    if i < 10:
        ref_image = skimage.io.imread(f"./images/kadid10k/images/I0{i}.png")
    else:
        ref_image = skimage.io.imread(f"./images/kadid10k/images/I{i}.png")
    reference_images.append(ref_image)



time_start = timeit.default_timer()

for j in range(1, NUMBER_OF_REFERENCE_IMAGES + 1):
    print(f"Collection no.: {j}")
    if j < 10:
        image_collection = skimage.io.imread_collection(f"./images/kadid10k/images/I0{j}_*.png", conserve_memory=True)
    else:
        image_collection = skimage.io.imread_collection(f"./images/kadid10k/images/I{j}_*.png", conserve_memory=True)

    mse_v, psnr_v, ssim_v, sg_essim_v = utils.iterate_images(reference_images[j - 1], image_collection, console_log=True)
    mse_values.extend(mse_v)
    psnr_values.extend(psnr_v)
    ssim_values.extend(ssim_v)
    sg_essim_values.extend(sg_essim_v)

time_end = timeit.default_timer()
print(f"Time elapsed for processing: {time_end - time_start:.2f} seconds\n")



plcc, srocc, krocc, rmse = utils.get_coefficients(dmos_values, sg_essim_values)

print(f"=======================================================\n")
print(f"=======================SG-ESSIM========================\n")
print(f"=======================================================\n")

print(f"\n===================\nPLCC: {plcc}\n===================\n")
print(f"\n===================\nSROCC: {srocc}\n===================\n")
print(f"\n===================\nKROCC: {krocc}\n===================\n")
print(f"\n===================\nRMSE: {rmse}\n===================\n")


plcc, srocc, krocc, rmse = utils.get_coefficients(dmos_values, ssim_values)

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
    "sg_essim": sg_essim_values
}

new_df = utils.save_values_to_df(dataframe, **value_dictionary)

print(f"Dataframe after iteration:\n {new_df.head(50)}\n\n")

new_df.to_csv("./result_kadid10k_iqa.csv", sep='\t', encoding='utf-8', index=False, header=True)


