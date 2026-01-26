from skimage.io import imread
from skimage.metrics import mean_squared_error
from ffs import FFS
from sgessim import SG_ESSIM
from lgv import LGV

from nits_db import NITS_DB
from kadid10k_db import KADID10K_DB
from tid2013_db import TID2013_DB
from tid2008_db import TID2008_DB
from htid_db import HTID_DB


def main():
    print("Main script started...\n")

    #########################################################################
    # Uncomment one of object from below to choose image database
    db = NITS_DB()

    # db = HTID_DB()

    # db = KADID10K_DB()

    # db = TID2013_DB()

    # db = TID2008_DB()
    #########################################################################


    # db.read_image_data()
    # db.load_reference_images()
    # db.load_deformed_image_collections()


    #########################################################################
    # Uncomment those two lines to test measures on selected image database
    db.load_images_and_data()
    db.perform_iqa()
    #########################################################################
    

    ### temp notes and tests:
    # sg_essim = SG_ESSIM()
    # ffs = FFS()
    # lgv = LGV()

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
        
    #     print("------------------------------------------------------")
    #     print(f"MSE: {mean_squared_error(ref_image, def_image)}")
    #     print(f"LGV: {lgv.calculate_lgv(ref_image, def_image)}")
    #     print("------------------------------------------------------\n")

    # print(f"MSE: {mean_squared_error(ref_image, def_image)}")
    # print(f"FFS: {ffs.calculate_ffs(ref_image, def_image)}")
    # print(f"SG_ESSIM: {sg_essim.calculate_sg_essim(ref_image, def_image)}")
    # print(f"LGV: {lgv.calculate_lgv(ref_image, def_image)}")




if __name__ == "__main__":
    main()

