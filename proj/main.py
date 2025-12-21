# from skimage.io import imread, show, imshow

from nits_db import NITS_DB
from kadid10k_db import KADID10K_DB
from csiq_db import CSIQ_DB


def main():
    print("Main script started...\n")

    # db = NITS_DB("database")
    # db.load_images_and_data()
    # db.perform_iqa("result_nits_iqa")


    # db = KADID10K_DB("database")
    # db.load_images_and_data()
    # db.perform_iqa("result_kadid10k_iqa")
    # db.read_image_data()


    # db = CSIQ_DB("database")
    # db.read_image_data()
    # db.load_reference_images()
    # db.load_deformed_image_collections()


    # ref_image = imread("./images/nits_iqa/Database/I1.bmp")
    # def_image = imread("./images/nits_iqa/Database/I1D8L5.bmp")

    # print(f"MSE: {mse(ref_image, def_image)}")

    # ref_image[200:800, 200:800, :] = [255, 0, 0]

    # imshow(ref_image)
    # show()

    # mat = scipy.io.loadmat('file.mat')



if __name__ == "__main__":
    main()

