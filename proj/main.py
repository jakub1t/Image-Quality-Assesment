from skimage.io import imread
from skimage.metrics import mean_squared_error
from ffs import calculate_ffs
from rsei import calculate_rsei
from sgessim import sg_essim

from nits_db import NITS_DB
from kadid10k_db import KADID10K_DB
from tid2013_db import TID2013_DB
from tid2008_db import TID2008_DB
from htid_db import HTID_DB


def main():
    print("Main script started...\n")

    # db = NITS_DB()

    db = HTID_DB()

    # db = KADID10K_DB()

    # db = TID2013_DB()

    # db = TID2008_DB()

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
    
    # ref_image = imread("./images/tid2008/reference_images/I01.bmp")
    # for i in range(1, 18):
    #     if i < 10:
    #         def_image = imread(f"./images/tid2008/distorted_images/I01_0{i}_4.bmp")
    #     else:
    #         def_image = imread(f"./images/tid2008/distorted_images/I01_{i}_4.bmp")

    # # print(f"MSE: {mean_squared_error(ref_image, def_image)}")
    # # print(f"FFS: {calculate_ffs(ref_image, def_image)}")
    # # print(f"SG_ESSIM: {sg_essim(ref_image, def_image)}")
    #     print(f"RSEI: {calculate_rsei(ref_image, def_image)}")




if __name__ == "__main__":
    main()

