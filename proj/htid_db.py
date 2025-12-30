
from pandas import DataFrame, Series
from scipy.io import loadmat
from skimage.io import imread, imread_collection

from iqa_manager import IQAManager
from image_data_loader import ImageDataLoader


class HTID_DB(IQAManager, ImageDataLoader):

    number_of_reference_images = 48
    
    def __init__(self, db_name: str = None):
        if db_name != None:
            self.db_name = db_name
        else: 
            self.db_name = "htid_iqa"


    def read_image_data(self):
        mat = loadmat("./images/htid/htid.mat")

        # mat to pandas df
        mat = {k:v for k, v in mat.items() if k[0] != '_'}
        self.df = DataFrame({k: Series(v[0]) for k, v in mat.items()})

        self.df["h_names"] = self.df["h_names"].apply(lambda x: str(x).replace("['", ""))
        self.df["h_names"] = self.df["h_names"].apply(lambda x: str(x).replace("']", ""))
        self.df["h_names"] = self.df["h_names"].apply(lambda x: str(x).replace("\\\\", "/"))

        self.df = self.df[~self.df["h_names"].str.contains("_01.png")]

        self.mos_values = self.df["h_mos"].values

        self.df.to_csv("./images/htid/htid.csv")

        print(self.df.head(50))
        print(" ")
    
    
    def load_reference_images(self):
        for i in range(1, self.number_of_reference_images + 1):
            if i < 10:
                ref_image = imread(f"./images/htid/set0{i}/im00{i}_01.png")
            else:
                ref_image = imread(f"./images/htid/set{i}/im0{i}_01.png")
            self.reference_images.append(ref_image)

        # for img in self.reference_images:
        #     print(img)
    

    def load_deformed_image_collections(self):
        temp_list = []
        for j in range(1, self.number_of_reference_images + 1):
            if j < 10:
                image_collection = imread_collection(f"./images/htid/set0{j}/*", conserve_memory=True)
            else:
                image_collection = imread_collection(f"./images/htid/set{j}/*", conserve_memory=True)
            temp_list.append(image_collection)

            # This thing below to remove first image from collections - its used for reference
            collection_as_list = image_collection.files
            collection_as_list.pop(0)
            # Reread collection - bleh
            new_collection = imread_collection(collection_as_list, conserve_memory=True)
            self.deformed_image_collections.append(new_collection)

        # for coll in self.deformed_image_collections:
        #     print(coll)

    
