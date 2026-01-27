
import numpy as np
from timeit import default_timer
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pandas import Series, DataFrame
from pandas import concat as pdconcat

from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy.optimize import curve_fit

from utils import logistic_regression_fun, safe_clip_nonfinite

from quality_measure import MSE, PSNR, SSIM
from sgessim import SG_ESSIM
from ffs import FFS
from lgv import LGV


class IQAManager:

    number_of_reference_images = 9
    number_of_processes = 1

    df = None
    mos_values = []
    reference_images = []
    deformed_image_collections = []


    mse = MSE("mse")
    psnr = PSNR("psnr")
    ssim = SSIM("ssim")
    sg_essim = SG_ESSIM("sg_essim")
    ffs = FFS("ffs")
    lgv = LGV("lgv")
    quality_measures = [mse, psnr, ssim, sg_essim, ffs, lgv]


    def __init__(self, db_name: str):
        self.db_name = db_name


    def calculate_quality_values(self):

        time_start = default_timer()

        for i, image_collection in enumerate(self.deformed_image_collections):

            measures_matrix, times_matrix = self.iterate_images(self.reference_images[i], image_collection, console_log=False)

            for j, measure in enumerate(self.quality_measures):
                measure.collected_values.extend(measures_matrix[j])
                measure.time_values.extend(times_matrix[j])

        time_end = default_timer()
        print(f"\nTime elapsed for processing: {time_end - time_start:.2f} seconds\n")

        for measure in self.quality_measures:
            measure.average_time = np.mean(measure.time_values)
            print(f">>> {measure.name.upper()} average time execution: {measure.average_time} seconds <<<")

        print("\n")


    def calculate_quality_from_measures(self, reference_image, image):

        times_list = []
        quality_values = []

        for measure in self.quality_measures:
            time_start = default_timer()
            quality_values.append(measure.calculate_quality(reference_image, image))
            time_end = default_timer()
            times_list.append(time_end - time_start)

        return quality_values, times_list


    def iterate_images(self, reference_image, image_array, console_log=False):
        
        image_list = image_array.files

        value_matrix = [[0 for x in range(1)] for y in range(len(self.quality_measures))]
        times_matrix = [[0 for x in range(1)] for y in range(len(self.quality_measures))]  

        with ProcessPoolExecutor() as executor:
            for i, results in enumerate(executor.map(self.calculate_quality_from_measures, repeat(reference_image), image_array)):

                times_list = list(results).pop()
                for t, time in enumerate(times_list):
                    times_matrix[t].append(time)

                # results = results[:-1]
                results = results[0]
                for j, result in enumerate(results):
                    value_matrix[j].append(result)

                if console_log == True:
                    # print(results)
                    print(f"Image: {image_list[i]}")
                    for k, measure in enumerate(self.quality_measures):
                        print(f"{measure.name}: {results[k]}")
                
                    print("\n")
                    
        value_matrix = [v_list[1:] for v_list in value_matrix]
        times_matrix = [v_list[1:] for v_list in times_matrix]

        return value_matrix, times_matrix


    def calculate_coefficients(self):

        dfs_cc_list = []

        for measure in self.quality_measures:
            
            plcc, srocc, krocc = self.get_coefficients(self.mos_values, measure.collected_values)

            print(f"=======================================================")
            print(f"======================={measure.name}========================")
            print(f"=======================================================\n")

            print(f"===================\nPLCC: {plcc}\n===================\n")
            print(f"===================\nSROCC: {srocc}\n===================\n")
            print(f"===================\nKROCC: {krocc}\n===================\n")

            dfs_cc_list.append(DataFrame(data=[plcc, srocc, krocc, np.round(measure.average_time, 6)], columns=[measure.name], index=["plcc", "srocc", "krocc", "average time execution [s]"]))

        dfs_cc = pdconcat(dfs_cc_list, axis=1)

        dfs_cc.to_csv(f"./csv_results/cc_result_{self.db_name}.csv", sep='\t', encoding='utf-8', header=True)


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


    def save_to_csv(self, df, csv_name):

        quality_measures_dictionary = {}

        for measure in self.quality_measures:
            quality_measures_dictionary[measure.name] = measure.collected_values

        new_df = self.save_values_to_df(df, **quality_measures_dictionary)

        print(f"Dataframe after iteration:\n {new_df.head(50)}\n\n")

        new_df.to_csv(f"./csv_results/{csv_name}.csv", sep='\t', encoding='utf-8', index=False, header=True)
    

    def perform_iqa(self, csv_name=""):
        self.calculate_quality_values()
        self.calculate_coefficients()
        if csv_name == "":
            self.save_to_csv(df=self.df, csv_name=f"result_{self.db_name}")
        else:
            self.save_to_csv(df=self.df, csv_name=csv_name)


