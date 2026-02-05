from iqa import IQA

from nits_db import NITS_DB
from kadid10k_db import KADID10K_DB
from tid2013_db import TID2013_DB
from tid2008_db import TID2008_DB
from htid_db import HTID_DB


def main():
    """Main function for the program with infinite loop."""
    
    image_databases = [NITS_DB(), HTID_DB(), KADID10K_DB(), TID2013_DB(), TID2008_DB()]
    num_of_available_dbs = len(image_databases)

    while True:
        correctly_selected = False
        
        print("Main script started...")
        while correctly_selected == False:
            print("\nSelect database that will be used for iqa:")
            print("Option no.: 0 -> exit")
            for i, img_db in enumerate(image_databases):
                print(f"Option no.: {i + 1} -> {img_db.db_name[:-4].upper()}")
            choice = input(f"Choose one of the numbers from 0 to {num_of_available_dbs} to select an option: ")
            print("\n")

            try:
                index = int(choice)
                if index <= num_of_available_dbs and index >= 0:
                    correctly_selected = True
                else:
                    correctly_selected = False
                    print("Incorrect option input...")
            except ValueError:
                print("Incorrect option input...")
                correctly_selected = False
        
        if index == 0:
            exit(0)
        else:
            executor = IQA(image_databases[index - 1])
            executor.run_iqa()

            # To run iqa with all image databases
            # executor = IQA(image_databases[0])
            # executor.run_iqa()
            # executor.set_image_database(image_databases[1])
            # executor.run_iqa()
            # executor.set_image_database(image_databases[2])
            # executor.run_iqa()
            # executor.set_image_database(image_databases[3])
            # executor.run_iqa()
            # executor.set_image_database(image_databases[4])
            # executor.run_iqa()



if __name__ == "__main__":
    main()

