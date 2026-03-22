import csv
import os

import pandas as pd

from .model import Result

DEFAULT_COLOR = '\033[39m'
RED_COLOR = "\033[31m"
YELLOW_COLOR = '\033[33m'

class ResultsWritter:
    @staticmethod
    def _ensure_dir(base_dir):
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
            print(f"{YELLOW_COLOR}Created directory '{base_dir}'{DEFAULT_COLOR}")
        elif not os.path.isdir(base_dir):
            raise NotADirectoryError(f"{RED_COLOR}{base_dir} is not a directory!{DEFAULT_COLOR}")
            
    @staticmethod
    def write_result(result: Result, file_name: str, base_dir: str):
        ResultsWritter._ensure_dir(base_dir)
        df = pd.DataFrame([result.to_dict()])
        path = os.path.join(base_dir, file_name if file_name.endswith('.csv') else f"{file_name}.csv")
        write_header = not os.path.exists(path) or os.path.getsize(path) == 0
        df.to_csv(path, index=False, mode= 'a', header=write_header)

    @staticmethod
    def write_dataframe(df: pd.DataFrame, file_name: str,  base_dir: str, replace=True):
        ResultsWritter._ensure_dir(base_dir)
        file_name = file_name if file_name.endswith('.csv') else f"{file_name}.csv"
        path_to_save = os.path.join(base_dir, file_name)
        df.to_csv(path_to_save, index=False, mode='w' if replace else 'a', header=replace)
