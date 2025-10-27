
import numpy as np
import skimage
from matplotlib import pyplot as plt
import measures as ms


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


# print(ms.mse(my_image, my_image))
# print(ms.rmse(my_image, my_image))
# print(ms.psnr(my_image, my_image))


original_image = skimage.io.imread("./images/nits_iqa/Database/I1.bmp")


image_array = []

for i in range(1, 6):
    image_array.append(skimage.io.imread(f"./images/nits_iqa/Database/I1D1L{i}.bmp"))

print(f"\nOriginal image shape: {original_image.shape}\n")

# plt.figure()

# f, axarr = plt.subplots(1, image_array.__len__() + 1) 
# axarr[0].imshow(original_image)

for ind, image in enumerate(image_array):
    print(f"Image no. {ind + 1}: ")
    # axarr[ind + 1].imshow(image)
    print("MSE: " + str(ms.mse(original_image, image)))
    print("RMSE: " + str(ms.rmse(original_image, image)))
    print("PSNR: " + str(ms.psnr(original_image, image)))
    print("SSIM: " + str(skimage.metrics.structural_similarity(original_image, image_array[ind], channel_axis=2)))

# plt.show()


print(ms.ssim(original_image, image_array[0]))


