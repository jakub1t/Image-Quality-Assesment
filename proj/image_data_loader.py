from abc import ABC
from abc import abstractmethod
from iqa_manager import IQAManager


class ImageDataLoader(ABC, IQAManager):

    @abstractmethod
    def read_image_data(self):
        """Read and preprocess (if necessary) data (MOS/DMOS) for images from image database."""
        pass
    
    
    @abstractmethod
    def load_reference_images(self):
        """
        Read reference images from image database into array field.
        """
        pass


    @abstractmethod
    def load_deformed_image_collections(self):
        """
        Method to read deformed images from image database into array field that contains image collections.
        """
        pass


    def load_images_and_data(self):
        """
        Method to execute all other abstract methods:
        - read_image_data, 
        - load_reference_images, 
        - load_deformed_image_collections.
        """
        self.read_image_data()
        self.load_reference_images()
        self.load_deformed_image_collections()
    
        