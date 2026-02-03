from abc import ABC
from abc import abstractmethod
from iqa_manager import IQAManager


class ImageDataLoader(ABC, IQAManager):
    """Abstract class that inherits core functionality of the program, allows to create separate ways to load an image database.

    Args:
        ABC: Python abc module helper class that provides a standard way to create an abstract class using inheritance.
        IQAManager: Class with the core program functionality
    """

    @abstractmethod
    def read_image_data(self):
        """Read and preprocess (if necessary) subjective scores (MOS/DMOS) for images from image database."""
        pass
    
    
    @abstractmethod
    def load_reference_images(self):
        """
        Read reference images from image database into list field.
        """
        pass


    @abstractmethod
    def load_deformed_image_collections(self):
        """
        Method to read deformed images from image database into list field that contains image collections.
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
    
        