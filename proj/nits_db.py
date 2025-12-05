
from pandas import read_excel
from skimage.io import imread, imread_collection
from timeit import default_timer

from utils import iterate_images

from image_database import ImageDatabase


class NITS_DB(ImageDatabase):

    number_of_reference_images = 9
    
    def __init__(self, db_name: str):
        self.db_name = db_name


    def read_image_data(self):
        self.df = read_excel("./images/nits_iqa/Database/Score.xlsx", sheet_name="Sheet1")
        self.df = self.df.sort_values(by=["Original Image Name", "Distorted Image Name"])
        self.df = self.df.reset_index(drop=True)

        self.mos_values = self.df["Score"].values
    
    
    def read_images(self):
        for i in range(1, self.number_of_reference_images + 1):
            ref_image = imread(f"./images/nits_iqa/Database/I{i}.bmp")
            self.reference_images.append(ref_image)


    def calculate_quality_values(self):
        time_start = default_timer()

        for j in range(1, self.number_of_reference_images + 1):
            print(f"Collection no.: {j}")
            image_collection = imread_collection(f"./images/nits_iqa/Database/I{j}D*.bmp", conserve_memory=True)

            mse_v, psnr_v, ssim_v, sg_essim_v, ffs_v = iterate_images(self.reference_images[j - 1], image_collection, console_log=True)
            self.mse_values.extend(mse_v)
            self.psnr_values.extend(psnr_v)
            self.ssim_values.extend(ssim_v)
            self.sg_essim_values.extend(sg_essim_v)
            self.ffs_values.extend(ffs_v)

        time_end = default_timer()
        print(f"Time elapsed for processing: {time_end - time_start:.2f} seconds\n")
