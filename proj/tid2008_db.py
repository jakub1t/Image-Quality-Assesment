import os
from skimage.io import imread_collection, imread
from pandas import DataFrame

from image_data_loader import ImageDataLoader


class TID2008_DB(ImageDataLoader):
    """TID2008 image database object that implements ImageDataLoader abstract class.
    Overrides three parent methods:
    - read_image_data
    - load_reference_images
    - load_deformed_image_collections

    Args:
        ImageDataLoader : Abstract parent class with the core functionality.
    """

    number_of_reference_images = 25
    reference_image_names = []
    
    def __init__(self, db_name: str = None):
        """Initializing method that allows to assign image database name used in process of naming the result csv files.

        Args:
            db_name (str, optional): Image database name to assign and to customize csv file names. Defaults to "tid2008_iqa".
        """
        if db_name != None:
            self.db_name = db_name
        else: 
            self.db_name = "tid2008_iqa"


    def read_image_data(self):
        """Overriden parent class. Loads MOS scores from txt file, saves them to df (Pandas DataFrame object) field 
        and creates structure for df field."""
        mos_col = []
        image_col = []

        with open("./images/tid2008/mos_with_names.txt") as mos_file:
            for line in mos_file:
                line = line.split()
                if line:
                    mos_col.append(line[0])
                    image_col.append(line[1])

        self.reference_image_names = image_col
        self.mos_values = mos_col

        self.df = DataFrame({"image_name":image_col, "mos":mos_col})
        self.normalize_image_names()
    
    
    def load_reference_images(self):
        """Overriden parent class. Saves loaded reference images in reference_images field."""
        temp_list = []
        for i in range(1, self.number_of_reference_images + 1):
            if i < 10:
                ref_image = imread(f"./images/tid2008/reference_images/i0{i}.bmp")
            else:
                ref_image = imread(f"./images/tid2008/reference_images/i{i}.bmp")
            temp_list.append(ref_image)
        self.reference_images[:] = temp_list

    
    def load_deformed_image_collections(self):
        """Overriden parent class. Saves loaded distorted image collections in deformed_image_collections field."""
        temp_list = []
        for j in range(1, self.number_of_reference_images + 1):
            if j < 10:
                image_collection = imread_collection(f"./images/tid2008/distorted_images/I0{j}_*", conserve_memory=True)
            else:
                image_collection = imread_collection(f"./images/tid2008/distorted_images/I{j}_*", conserve_memory=True)
            temp_list.append(image_collection)
        self.deformed_image_collections[:] = temp_list


    def normalize_image_names(self):
        """Method that changes all file names to lower case, used to eliminate irregularities in file names that are present in TID2008.
        """
        for path, dirs, files in os.walk("./images/tid2008/distorted_images"):
            for file in files:
                new_file = file.lower()
                os.rename(os.path.join(path, file), os.path.join(path, new_file))
        for path, dirs, files in os.walk("./images/tid2008/reference_images"):
            for file in files:
                new_file = file.lower()
                os.rename(os.path.join(path, file), os.path.join(path, new_file))
    
