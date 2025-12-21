# from skimage.io import imread
# from iqa_manager import mse

from nits_db import NITS_DB
from kadid10k_db import KADID10K_DB
from csiq_db import CSIQ_DB
from tid2013_db import TID2013_DB


def main():
    print("Main script started...\n")

    # db = NITS_DB("database")

    # db = KADID10K_DB("database")

    # db = TID2013_DB("database")

    # db = CSIQ_DB("database")

    # db.read_image_data()
    # db.load_reference_images()
    # db.load_deformed_image_collections()

    # db.load_images_and_data()
    # db.perform_iqa("result_nits_iqa")
    # db.perform_iqa("result_kadid10k_iqa")
    # db.perform_iqa("result_tid2013_iqa")



    ### temp notes and tests:

    # ref_image = imread("./images/nits_iqa/Database/I1.bmp")
    # def_image = imread("./images/nits_iqa/Database/I1D8L5.bmp")

    # ref_image = imread("./images/tid2013/reference_images/I01.bmp")
    # def_image = imread("./images/tid2013/distorted_images/I01_01_2.bmp")

    # print(f"MSE: {mse(ref_image, def_image)}")

    # mat = scipy.io.loadmat('file.mat')



if __name__ == "__main__":
    main()

