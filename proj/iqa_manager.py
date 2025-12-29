
import numpy as np
from timeit import default_timer
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pandas import Series

from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy.optimize import curve_fit
from skimage.metrics import structural_similarity, mean_squared_error, peak_signal_noise_ratio

from utils import logistic_regression_fun, safe_clip_nonfinite
from sgessim import sg_essim
from ffs import calculate_ffs
from rsei import calculate_rsei


class IQAManager:

    number_of_reference_images = 9
    number_of_processes = 1

    df = None
    mos_values = []
    reference_images = []
    deformed_image_collections = []

    mse_values = []
    psnr_values = [] 
    ssim_values = []
    sg_essim_values = []
    ffs_values = []
    rsei_values = []

    quality_measures_dictionary = {
        "mse": mse_values,
        "psnr": psnr_values,
        "ssim": ssim_values,
        "sg_essim": sg_essim_values,
        "ffs": ffs_values,
        "rsei": rsei_values
    }


    def __init__(self, db_name: str):
        self.db_name = db_name


    def calculate_quality_values(self):

        time_start = default_timer()

        for j, image_collection in enumerate(self.deformed_image_collections):

            mse_v, psnr_v, ssim_v, sg_essim_v, ffs_v, rsei_v = self.iterate_images(self.reference_images[j], image_collection, console_log=True)
            self.mse_values.extend(mse_v)
            self.psnr_values.extend(psnr_v)
            self.ssim_values.extend(ssim_v)
            self.sg_essim_values.extend(sg_essim_v)
            self.ffs_values.extend(ffs_v)
            self.rsei_values.extend(rsei_v)

        time_end = default_timer()
        print(f"\nTime elapsed for processing: {time_end - time_start:.2f} seconds\n")


    def calculate_quality_from_measures(reference_image, image):
            
        mse_val = mean_squared_error(reference_image, image)

        psnr_val = peak_signal_noise_ratio(reference_image, image)

        ssim_val = structural_similarity(reference_image, image, channel_axis=2)

        sg_essim_val = sg_essim(reference_image, image)

        ffs_val = calculate_ffs(reference_image, image)

        rsei_val = calculate_rsei(reference_image, image)
            

        return mse_val, psnr_val, ssim_val, sg_essim_val, ffs_val, rsei_val


    def iterate_images(self, reference_image, image_array, console_log=False):
        
        image_list = image_array.files
        
        mse_list = []
        psnr_list = []
        ssim_list = []
        sg_essim_list = []
        ffs_list = []
        rsei_list = []

        with ProcessPoolExecutor() as executor:
            for i, result in enumerate(executor.map(IQAManager.calculate_quality_from_measures, repeat(reference_image), image_array)):
                mse_val, psnr_val, ssim_val, sg_essim_val, ffs_val, rsei_val = result
                mse_list.append(mse_val)
                psnr_list.append(psnr_val)
                ssim_list.append(ssim_val)
                sg_essim_list.append(sg_essim_val)
                ffs_list.append(ffs_val)
                rsei_list.append(rsei_val)

                if console_log == True:
                    # print(result)
                    print(f"Image: {image_list[i]}")
                    print(f"MSE: {mse_val}")
                    print(f"PSNR: {psnr_val}")
                    print(f"SSIM: {ssim_val}")
                    print(f"SG-ESSIM: {sg_essim_val}")
                    print(f"FFS: {ffs_val}")
                    print(f"RSEI: {rsei_val}")
                
                    print("\n")

        return mse_list, psnr_list, ssim_list, sg_essim_list, ffs_list, rsei_list


    def calculate_coefficients(self):

        for quality_name, quality_values in self.quality_measures_dictionary.items():
            
            plcc, srocc, krocc = self.get_coefficients(self.mos_values, quality_values)

            print(f"=======================================================")
            print(f"======================={quality_name}========================")
            print(f"=======================================================\n")

            print(f"===================\nPLCC: {plcc}\n===================\n")
            print(f"===================\nSROCC: {srocc}\n===================\n")
            print(f"===================\nKROCC: {krocc}\n===================\n")


    def get_coefficients(self, x, y):
        # Non-linear regression for PLCC
        x = safe_clip_nonfinite(x)
        y = safe_clip_nonfinite(y)
        b1_0 = (max(y) - min(y))
        initial_guess = [b1_0, 1, np.mean(x), 0, np.mean(y)]
        betas, _ = curve_fit(logistic_regression_fun, x, y, p0=initial_guess, maxfev=20000)
        x_mapped = logistic_regression_fun(x, *betas)

        # Pearson’s linear correlation coefficient
        pearson_coefficient, _ = pearsonr(x_mapped, y)

        pearson_coefficient = np.abs(np.round(pearson_coefficient, 3))

        # Spearman’s rank-order correlation coefficient 
        spearman_coefficient, _ = spearmanr(x, y)

        spearman_coefficient = np.abs(np.round(spearman_coefficient, 3))

        # Kendall’s rank order correlation coefficient
        kendall_coefficient, _ = kendalltau(x, y)

        kendall_coefficient = np.abs(np.round(kendall_coefficient, 3))

        return pearson_coefficient, spearman_coefficient, kendall_coefficient


    def save_values_to_df(self, df, **kwargs):

        df.insert(len(df.columns), " ", " ", False)

        for key, value in kwargs.items():
            df.insert(len(df.columns), key, "NaN", False)
            df[key] = Series(value)

        return df


    def save_to_csv(self, csv_name):

        new_df = self.save_values_to_df(self.df, **self.quality_measures_dictionary)

        print(f"Dataframe after iteration:\n {new_df.head(50)}\n\n")

        new_df.to_csv(f"./{csv_name}.csv", sep='\t', encoding='utf-8', index=False, header=True)
    

    def perform_iqa(self, csv_name=""):
        self.calculate_quality_values()
        self.calculate_coefficients()
        if csv_name == "":
            self.save_to_csv(csv_name=f"result_{self.db_name}")
        else:
            self.save_to_csv(csv_name=csv_name)


