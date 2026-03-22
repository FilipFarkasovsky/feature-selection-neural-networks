import os

import pandas as pd

from feature_selectors.base_models import ResultType
from data.sampling import SamplingType

DEFAULT_COLOR = '\033[39m'
RED_COLOR = "\033[31m"
CYAN_COLOR = '\033[36m'
GREEN_COLOR = '\033[32m'
YELLOW_COLOR = '\033[33m'

class ResultsLoader:
    @staticmethod
    def load_all(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"'{path}' does not exist!")

        if os.path.isdir(path):
            csv_files = [f for f in ResultsLoader._files_in_dir_tree(path) if f.endswith('.csv')]
            if not csv_files:
                raise FileNotFoundError(f"No csv results found at {path}!")
            return pd.concat((pd.read_csv(p) for p in csv_files), ignore_index=True)
        elif path.endswith('.csv'):
            return pd.read_csv(path)
        else:
            raise ValueError(f"{RED_COLOR}{path} must be a CSV file or a directory containing CSVs.{DEFAULT_COLOR}", RED_COLOR)
        
    @staticmethod
    def load_by(path, field, value, allowed_values=None):
        if allowed_values is not None and value not in allowed_values:
            raise ValueError(f"{RED_COLOR}{value} must be one of {allowed_values}{DEFAULT_COLOR}")
        df = ResultsLoader.load_all(path)
        filtered = df[df[field] == value].reset_index(drop=True)
        if filtered.empty:
            raise ValueError(f"{YELLOW_COLOR}No results found for {field} = {value}{DEFAULT_COLOR}")
        return filtered

    @staticmethod
    def load_by_sampling(results_path, sampling):
        sampling_types = {s.value for s in SamplingType}
        return ResultsLoader.load_by(results_path,'sampling', sampling, allowed_values=sampling_types)

    @staticmethod
    def load_by_result_type(results_path, result_type):
        result_types = {rt.value for rt in ResultType}
        return ResultsLoader.load_by(results_path, 'result_type', result_type, allowed_values=result_types)
    
    @staticmethod    
    def load_by_dataset(results_path, dataset_name):
        return ResultsLoader.load_by(results_path, 'dataset_name', dataset_name)

    @staticmethod    
    def load_by_name(results_path, name):
        return ResultsLoader.load_by(results_path, 'name', name)

    @staticmethod    
    def _files_in_dir_tree(path):
        if not os.path.isdir(path):
            raise ValueError(f"{path} is not a directory!")

        return [
            os.path.join(root, name)
            for root, _, files in os.walk(path, topdown=False)
            for name in files
        ]