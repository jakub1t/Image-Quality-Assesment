
from image_data_loader import ImageDataLoader


class IQA:

    def __init__(self, image_db: ImageDataLoader):
        self.image_db = image_db


    def set_image_database(self, image_db: ImageDataLoader):
        self.image_db = image_db


    def run_iqa(self):
        self.image_db.load_images_and_data()
        self.image_db.perform_iqa()


