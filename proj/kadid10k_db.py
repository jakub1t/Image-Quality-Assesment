
from pandas import read_csv
from skimage.io import imread, imread_collection

from image_data_loader import ImageDataLoader


class KADID10K_DB(ImageDataLoader):
    """KADID-10K image database object that implements ImageDataLoader abstract class.
    Overrides three parent methods:\n
    - read_image_data,\n
    - load_reference_images,\n
    - load_deformed_image_collections.\n

    Args:
        ImageDataLoader : Abstract parent class with the core functionality.
    """

    number_of_reference_images = 81
    
    def __init__(self, db_name: str = None):
        """Initializing method that allows to assign image database name used in process of naming the result csv files.

        Args:
            db_name (str, optional): Image database name to assign and to customize csv file names. Defaults to "kadid10k_iqa".
        """
        if db_name != None:
            self.db_name = db_name
        else: 
            self.db_name = "kadid10k_iqa"


    def read_image_data(self):
        """Overriden parent class. Loads DMOS scores from csv file and saves them to df (Pandas DataFrame object) field."""
        self.df = read_csv("./images/kadid10k/dmos.csv", sep=",")

        self.mos_values = self.df["dmos"].values
    
    
    def load_reference_images(self):
        """Overriden parent class. Saves loaded reference images in reference_images field."""
        temp_list = []
        for i in range(1, self.number_of_reference_images + 1):
            if i < 10:
                ref_image = imread(f"./images/kadid10k/images/I0{i}.png")
            else:
                ref_image = imread(f"./images/kadid10k/images/I{i}.png")
            temp_list.append(ref_image)
        self.reference_images[:] = temp_list


    def load_deformed_image_collections(self):
        """Overriden parent class. Saves loaded distorted image collections in deformed_image_collections field."""
        temp_list = []
        for j in range(1, self.number_of_reference_images + 1):
            if j < 10:
                image_collection = imread_collection(f"./images/kadid10k/images/I0{j}_*.png", conserve_memory=True)
            else:
                image_collection = imread_collection(f"./images/kadid10k/images/I{j}_*.png", conserve_memory=True)
            temp_list.append(image_collection)
        self.deformed_image_collections[:] = temp_list

