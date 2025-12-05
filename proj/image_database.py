

from utils import save_values_to_df, get_coefficients


class ImageDatabase:

    number_of_reference_images = 9
    df = None
    mos_values = []
    reference_images = []

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
    
    
    def read_images(self):
        print("Read reference images from database...")


    def calculate_quality_values(self):
        print("Read deformed images from database, measure time execution and calculate quality measures...")


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
        self.read_images()
        self.calculate_quality_values()
        self.calculate_coefficients()
        self.save_to_csv(csv_name=csv_name)




