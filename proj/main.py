
from nits_db import NITS_DB
from kadid10k_db import KADID10K_DB
from tid2013_db import TID2013_DB
from tid2008_db import TID2008_DB
from htid_db import HTID_DB


def main():
    print("Main script started...\n")

    #########################################################################

    # Leave one from lines below uncommented to choose image database

    db = NITS_DB()

    # db = HTID_DB()

    # db = KADID10K_DB()

    # db = TID2013_DB()

    # db = TID2008_DB()

    #########################################################################


    #########################################################################

    # Leave those two lines uncommented to test measures on selected image database

    db.load_images_and_data()
    db.perform_iqa()

    #########################################################################


if __name__ == "__main__":
    main()

