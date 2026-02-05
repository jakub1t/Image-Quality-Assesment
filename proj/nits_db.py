
from pandas import read_excel
from skimage.io import imread, imread_collection

from image_data_loader import ImageDataLoader


class NITS_DB(ImageDataLoader):
    """NITS image database object that implements ImageDataLoader abstract class.
    Overrides three parent methods:\n
    - read_image_data,\n
    - load_reference_images,\n
    - load_deformed_image_collections.\n

    Args:
        ImageDataLoader : Abstract parent class with the core functionality.
    """

    number_of_reference_images = 9
    
    def __init__(self, db_name: str = None):
        """Initializing method that allows to assign image database name used in process of naming the result csv files.

        Args:
            db_name (str, optional): Image database name to assign and to customize csv file names. Defaults to "nits_iqa".
        """
        if db_name != None:
            self.db_name = db_name
        else: 
            self.db_name = "nits_iqa"


    def read_image_data(self):
        """Overriden parent class. Loads MOS scores from excel file and saves them to df (Pandas DataFrame object) field."""
        self.df = read_excel("./images/nits_iqa/Database/Score.xlsx", sheet_name="Sheet1")
        self.df = self.df.sort_values(by=["Original Image Name", "Distorted Image Name"])
        self.df = self.df.reset_index(drop=True)

        self.mos_values = self.df["Score"].values
    
    
    def load_reference_images(self):
        """Overriden parent class. Saves loaded reference images in reference_images field."""
        temp_list = []
        for i in range(1, self.number_of_reference_images + 1):
            ref_image = imread(f"./images/nits_iqa/Database/I{i}.bmp")
            temp_list.append(ref_image)
        self.reference_images[:] = temp_list

    
    def load_deformed_image_collections(self):
        """Overriden parent class. Saves loaded distorted image collections in deformed_image_collections field."""
        temp_list = []
        for j in range(1, self.number_of_reference_images + 1):
            image_collection = imread_collection(f"./images/nits_iqa/Database/I{j}D*.bmp", conserve_memory=True)
            temp_list.append(image_collection)
        self.deformed_image_collections[:] = temp_list



    
