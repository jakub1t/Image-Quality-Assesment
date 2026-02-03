
from image_data_loader import ImageDataLoader


class IQA:
    """Class that allows user to choose image database 
    by passing ImageDataLoader object.
    """

    def __init__(self, image_db: ImageDataLoader):
        """Initializing method that allows to assign ImageDataLoader object to image_db field
        and choose context in order to run the rest of the program with this context.

        Args:
            image_db (ImageDataLoader): ImageDataLoader object to assign.
        """
        self.image_db = image_db


    def set_image_database(self, image_db: ImageDataLoader):
        """Method that allows to assign ImageDataLoader object to image_db field
        and choose context during runtime in order to run the rest of the program with this context.

        Args:
            image_db (ImageDataLoader): ImageDataLoader object to assign.
        """
        self.image_db = image_db


    def run_iqa(self):
        """Method that is used to run the rest of the program 
        based on passed context in image_db field.
        """
        self.image_db.load_images_and_data()
        self.image_db.perform_iqa()


