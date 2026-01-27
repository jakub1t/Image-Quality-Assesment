import os
from skimage.io import imread_collection, imread
from pandas import DataFrame

from iqa_manager import IQAManager
from image_data_loader import ImageDataLoader


class TID2013_DB(IQAManager, ImageDataLoader):

    number_of_reference_images = 25
    reference_image_names = []
    
    def __init__(self, db_name: str = None):
        if db_name != None:
            self.db_name = db_name
        else: 
            self.db_name = "tid2013_iqa"


    def read_image_data(self):
        mos_col = []
        image_col = []

        with open("./images/tid2013/mos_with_names.txt") as mos_file:
            for line in mos_file:
                line = line.split()
                if line:
                    mos_col.append(line[0])
                    image_col.append(line[1])

        self.reference_image_names = image_col
        self.mos_values = mos_col

        self.df = DataFrame({"image_name":image_col, "mos":mos_col})
        self.change_names_in_distorted_images()
    
    
    def load_reference_images(self):
        # self.reference_images = imread_collection("./images/tid2013/reference_images/*", conserve_memory=True) # This line unfortunately does not work with ProcessPoolExecutor
        for i in range(1, self.number_of_reference_images + 1):
            if i < 10:
                ref_image = imread(f"./images/tid2013/reference_images/i0{i}.bmp")
            else:
                ref_image = imread(f"./images/tid2013/reference_images/i{i}.bmp")
            self.reference_images.append(ref_image)

    
    def load_deformed_image_collections(self):
        for j in range(1, self.number_of_reference_images + 1):
            if j < 10:
                image_collection = imread_collection(f"./images/tid2013/distorted_images/I0{j}_*", conserve_memory=True)
            else:
                image_collection = imread_collection(f"./images/tid2013/distorted_images/I{j}_*", conserve_memory=True)
            self.deformed_image_collections.append(image_collection)


    def change_names_in_distorted_images(self):
        for path, dirs, files in os.walk("./images/tid2013/distorted_images"):
            for file in files:
                new_file = file.lower()
                os.rename(os.path.join(path, file), os.path.join(path, new_file))
        for path, dirs, files in os.walk("./images/tid2013/reference_images"):
            for file in files:
                new_file = file.lower()
                os.rename(os.path.join(path, file), os.path.join(path, new_file))
    
