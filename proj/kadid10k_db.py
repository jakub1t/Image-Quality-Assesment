
from pandas import read_csv
from skimage.io import imread, imread_collection

from iqa_manager import IQAManager
from image_data_loader import ImageDataLoader


class KADID10K_DB(IQAManager, ImageDataLoader):

    number_of_reference_images = 81
    
    def __init__(self, db_name: str = None):
        if db_name != None:
            self.db_name = db_name
        else: 
            self.db_name = "kadid10k_iqa"


    def read_image_data(self):
        self.df = read_csv("./images/kadid10k/dmos.csv", sep=",")

        self.mos_values = self.df["dmos"].values
    
    
    def load_reference_images(self):
        for i in range(1, self.number_of_reference_images + 1):
            if i < 10:
                ref_image = imread(f"./images/kadid10k/images/I0{i}.png")
            else:
                ref_image = imread(f"./images/kadid10k/images/I{i}.png")
            self.reference_images.append(ref_image)


    def load_deformed_image_collections(self):
        for j in range(1, self.number_of_reference_images + 1):
            if j < 10:
                image_collection = imread_collection(f"./images/kadid10k/images/I0{j}_*.png", conserve_memory=True)
            else:
                image_collection = imread_collection(f"./images/kadid10k/images/I{j}_*.png", conserve_memory=True)
            self.deformed_image_collections.append(image_collection)

