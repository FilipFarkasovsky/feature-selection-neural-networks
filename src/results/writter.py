import csv
import os

import pandas as pd

from .model import Result

class ResultsWritter:
    @staticmethod
    def write_result(result: Result, file_name: str, base_dir: str):
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        elif not os.path.isdir(base_dir):
            raise Exception(f"base_dir[{base_dir}] must be a directory!")
        
        result_dict = result.to_dict()
        file_name = file_name if file_name.endswith('.csv') else f"{file_name}.csv"
        path_to_save = os.path.join(base_dir, file_name)
        with open(path_to_save, "a") as f:
            writer = csv.DictWriter(f, result_dict.keys())

            if f.tell() == 0:
                writer.writeheader()

            writer.writerow(result_dict)

    @staticmethod
    def write_dataframe(df: pd.DataFrame, file_name: str,  base_dir: str, replace=True):
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        elif not os.path.isdir(base_dir):
            raise Exception(f"base_dir[{base_dir}] must be a directory!")
        
        file_name = file_name if file_name.endswith('.csv') else f"{file_name}.csv"
        path_to_save = os.path.join(base_dir, file_name)

        if replace or not os.path.exists(path_to_save):
            df.to_csv(path_to_save, index=False)
        else:
            df.to_csv(path_to_save, mode='a', index=False, header=False)
