import os

import pandas as pd

from feature_selectors.base_models import ResultType
from data.sampling import SamplingType
from util.filesystem import files_in_dir_tree


class ResultsLoader:
    @staticmethod
    def load_all(results_path):
        if not os.path.exists(results_path):
            raise Exception(f"'{results_path}' does not exist!")

        is_dir = False
        if os.path.isdir(results_path):
            is_dir = True
            if [f for f in files_in_dir_tree(results_path) if f.endswith('.csv')] == []:
                raise Exception(f"No results found at {results_path}!")

        elif not results_path.endswith('.csv'):
            raise Exception(f"{results_path} should be either a path to dir "
                            f"containing csv files or a path to csv itself.")


        if is_dir:
            paths = files_in_dir_tree(results_path)
            return pd.concat((pd.read_csv(p) for p in paths), ignore_index=True)
        else:
            return pd.read_csv(results_path)
        
    @staticmethod
    def load_by(results_path, field, field_value, allowed_field_values=None):
        if allowed_field_values is not None and field_value not in allowed_field_values:
            raise Exception(f"{field_value} must be one of: {allowed_field_values}")

        df = ResultsLoader.load_all(results_path)
        results = df[df[field] == field_value].reset_index(drop=True)

        if results.size == 0:
            raise Exception(f"No results found for {field} {field_value}")

        return results

    @staticmethod
    def load_by_sampling(results_path, sampling):
        sampling_types = {s.value for s in SamplingType}
        return ResultsLoader.load_by(results_path,'sampling', sampling, allowed_field_values=sampling_types)

    @staticmethod
    def load_by_result_type(results_path, result_type):
        result_types = {rt.value for rt in ResultType}
        return ResultsLoader.load_by(results_path, 'result_type', result_type, allowed_field_values=result_types)
    
    @staticmethod    
    def load_by_dataset(results_path, dataset_name):
        return ResultsLoader.load_by(results_path, 'dataset_name', dataset_name)

    @staticmethod    
    def load_by_name(results_path, name):
        return ResultsLoader.load_by(results_path, 'name', name)
