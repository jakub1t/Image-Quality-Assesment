# from skimage.io import imread
# from skimage.metrics import mean_squared_error

from nits_db import NITS_DB
from kadid10k_db import KADID10K_DB
from csiq_db import CSIQ_DB
from tid2013_db import TID2013_DB
from htid_db import HTID_DB


def main():
    print("Main script started...\n")

    db = NITS_DB("nits_iqa")

    # db = HTID_DB("htid_iqa")

    # db = KADID10K_DB("kadid10k_iqa")

    # db = TID2013_DB("tid2013_iqa")

    # db = CSIQ_DB("database")

    # db.read_image_data()
    # db.load_reference_images()
    # db.load_deformed_image_collections()

    db.load_images_and_data()
    db.perform_iqa()



    ### temp notes and tests:

    # ref_image = imread("./images/nits_iqa/Database/I1.bmp")
    # def_image = imread("./images/nits_iqa/Database/I1D8L5.bmp")

    # ref_image = imread("./images/tid2013/reference_images/I01.bmp")
    # def_image = imread("./images/tid2013/distorted_images/I01_01_2.bmp")

    # print(f"MSE: {mean_squared_error(ref_image, def_image)}")




if __name__ == "__main__":
    main()

