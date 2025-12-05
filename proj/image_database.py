
from timeit import default_timer

from utils import save_values_to_df, get_coefficients, iterate_images


class ImageDatabase:

    number_of_reference_images = 9
    df = None
    mos_values = []
    reference_images = []
    deformed_image_collections = []

    mse_values = []
    psnr_values = [] 
    ssim_values = []
    sg_essim_values = []
    ffs_values = []

    quality_measures_dictionary = {
        "mse": mse_values,
        "psnr": psnr_values,
        "ssim": ssim_values,
        "sg_essim": sg_essim_values,
        "ffs": ffs_values
    }


    def __init__(self, db_name: str):
        self.db_name = db_name
    

    def read_image_data(self):
        print("Read data for database images...")
    
    
    def load_images(self):
        print("Read reference images from database...")


    def load_and_get_deformed_image_collections():
        print("Read deformed images from database")


    def calculate_quality_values(self):
        time_start = default_timer()

        for j, image_collection in enumerate(self.deformed_image_collections):

            mse_v, psnr_v, ssim_v, sg_essim_v, ffs_v = iterate_images(self.reference_images[j], image_collection, console_log=True)
            self.mse_values.extend(mse_v)
            self.psnr_values.extend(psnr_v)
            self.ssim_values.extend(ssim_v)
            self.sg_essim_values.extend(sg_essim_v)
            self.ffs_values.extend(ffs_v)

        time_end = default_timer()
        print(f"Time elapsed for processing: {time_end - time_start:.2f} seconds\n")


    def calculate_coefficients(self):

        for quality_name, quality_values in self.quality_measures_dictionary.items():

            plcc, srocc, krocc = get_coefficients(self.mos_values, quality_values)

            print(f"=======================================================")
            print(f"======================={quality_name}========================")
            print(f"=======================================================\n")

            print(f"===================\nPLCC: {plcc}\n===================\n")
            print(f"===================\nSROCC: {srocc}\n===================\n")
            print(f"===================\nKROCC: {krocc}\n===================\n")


    def save_to_csv(self, csv_name):

        new_df = save_values_to_df(self.df, **self.quality_measures_dictionary)

        print(f"Dataframe after iteration:\n {new_df.head(50)}\n\n")

        new_df.to_csv(f"./{csv_name}.csv", sep='\t', encoding='utf-8', index=False, header=True)
    

    def calculate_everything(self, csv_name):
        self.read_image_data()
        self.load_images()
        self.load_and_get_deformed_image_collections()
        self.calculate_quality_values()
        self.calculate_coefficients()
        self.save_to_csv(csv_name=csv_name)




