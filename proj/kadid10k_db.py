
from pandas import read_csv
from skimage.io import imread, imread_collection
from timeit import default_timer

from utils import iterate_images

from image_database import ImageDatabase


class KADID10K_DB(ImageDatabase):

    number_of_reference_images = 81
    
    def __init__(self, db_name: str):
        self.db_name = db_name


    def read_image_data(self):
        self.df = read_csv("./images/kadid10k/dmos.csv", sep=",")

        self.mos_values = self.df["dmos"].values
    
    
    def read_images(self):
        for i in range(1, self.number_of_reference_images + 1):
            if i < 10:
                ref_image = imread(f"./images/kadid10k/images/I0{i}.png")
            else:
                ref_image = imread(f"./images/kadid10k/images/I{i}.png")
            self.reference_images.append(ref_image)


    def calculate_quality_values(self):
        time_start = default_timer()

        for j in range(1, self.number_of_reference_images + 1):
            print(f"Collection no.: {j}")
            if j < 10:
                image_collection = imread_collection(f"./images/kadid10k/images/I0{j}_*.png", conserve_memory=True)
            else:
                image_collection = imread_collection(f"./images/kadid10k/images/I{j}_*.png", conserve_memory=True)

            mse_v, psnr_v, ssim_v, sg_essim_v, ffs_v = iterate_images(self.reference_images[j - 1], image_collection, console_log=True)
            self.mse_values.extend(mse_v)
            self.psnr_values.extend(psnr_v)
            self.ssim_values.extend(ssim_v)
            self.sg_essim_values.extend(sg_essim_v)
            self.ffs_values.extend(ffs_v)

        time_end = default_timer()
        print(f"Time elapsed for processing: {time_end - time_start:.2f} seconds\n")
