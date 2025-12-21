import os
from pandas import read_excel
from skimage.io import imread_collection, imread

from iqa_manager import IQAManager
from image_data_loader import ImageDataLoader


class CSIQ_DB(IQAManager, ImageDataLoader):

    number_of_reference_images = 30

    # reference_image_names = ["/1600*", "/aerial_city*", "/boston*", "/bridge*", "/butter*", 
    #                          "/cactus*", "/child*", "/couple*", "/elk*", "/family*",
    #                          "/fisher*", "/foxy*", "/geckos*", "/lady_liberty*", "/lake*", 
    #                          "/log*", "/monument*", "/native_american*", "/redwood*", "/roping*",
    #                          "/rushmore*", "/shroom*", "/snow_leaves*", "/sunset_sparrow*", "/sunsetcolor*", 
    #                          "/swarm*", "/trolley*", "/turtle*", "/veggies*", "/woman*"]
    reference_image_names = []
    
    def __init__(self, db_name: str):
        self.db_name = db_name


    def read_image_data(self):
        self.df = read_excel("./images/csiq/csiq.DMOS.xlsx", sheet_name="all_by_image", header=3, usecols="D:I")
        # self.df["dst_type"] = self.df["dst_type"].replace({"noise":"awgn"})
        # self.df = self.df.sort_values(by=["image", "dst_type"])
        # self.df = self.df.reset_index(drop=True)

        self.reference_image_names = self.df["image"].unique().astype(str)
        # self.reference_image_names = ["/" + s + "*" for s in self.reference_image_names]
        print(self.reference_image_names)

        print(self.df.head(50))

        self.mos_values = self.df["dmos"].values
    
    
    def load_reference_images(self):
        self.reference_images = imread_collection("./images/csiq/src_imgs/*", conserve_memory=True)
        # references = self.reference_images.files
        # for reference in references:
        #     print(reference)

    
    def load_deformed_image_collections(self):
        root_dir = "./images/csiq/dst_imgs/"
        temp = []

        image_directories = imread_collection(root_dir + "*", conserve_memory=True)
        image_directories_names = image_directories.files
        print(image_directories_names)
        
        for subdir, dirs, files in os.walk(root_dir):
            for file in files:
                print(file)
                # print(os.path.join(subdir, file))


        # for name in self.reference_image_names:
        #     temp = [s + name for s in image_directories_names]
        #     collection = imread_collection(temp, conserve_memory=True)
        #     for file in collection.files:
        #         if file == f"./images/csiq/dst_imgs/contrast/{name}.contrast.5.png":
        #             collection.files.remove(file)
            
        #     self.deformed_image_collections.append(collection)

        # for i, collec in enumerate(self.deformed_image_collections):
        #     print(f"Collection No.: {i + 1}")
        #     for file in collec.files:
        #         print(file)

        # collection = imread_collection(["./images/csiq/dst_imgs/awgn/*", 
        #                                 "./images/csiq/dst_imgs/blur/*", 
        #                                 "./images/csiq/dst_imgs/contrast/*",
        #                                 "./images/csiq/dst_imgs/fnoise/*",
        #                                 "./images/csiq/dst_imgs/jpeg/*",
        #                                 "./images/csiq/dst_imgs/jpeg2000/*"], conserve_memory=True)
        

    def file_sort():
        pass