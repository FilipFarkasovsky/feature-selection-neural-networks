import json

import numpy as np
import pandas as pd

from util.dict import flatten_dict
from results.loader import ResultsLoader
from evaluation.selection import SelectionScorer


DEFAULT_COLOR = '\033[39m'
CYAN_COLOR = '\033[36m'
GREEN_COLOR = '\033[32m'
YELLOW_COLOR = '\033[33m'
WHITE_COLOR = '\033[37m'


class ResultsScorer:
    DEFAULT_EVALUATE_AT = [5, 10, 20, 50, 100, 200]
    DEFAULT_VERBOSE = 1

    @staticmethod
    def score_all(datasets, results_path, evaluate_at = DEFAULT_EVALUATE_AT):
        scores = []

        try:
            subset_results = ResultsLoader.load_by_result_type(results_path, 'subset')
            subset_scores = ResultsScorer.evaluate_subsets(datasets, subset_results)
            scores.append(subset_scores)
        except Exception as e:
            print(f"Could not load subset results, reson: {e}")

        try:
            rank_results = ResultsLoader.load_by_result_type(results_path, 'rank')
            rank_scores = ResultsScorer.evaluate_ordered(datasets, rank_results, evaluate_at)
            scores.append(rank_scores)
        except Exception as e:
            print(f"Could not load rank results, reson: {e}")

        try:
            weights_results = ResultsLoader.load_by_result_type(results_path, 'weights')
            weights_scores = ResultsScorer.evaluate_ordered(datasets, weights_results, evaluate_at)
            scores.append(weights_scores)
        except Exception as e:
            print(f"Could not load weighted results, reson: {e}")

        if len(scores) == 0:
            raise Exception("Could generate any scoring for given results.")

        return pd.concat(scores)

    @staticmethod
    def _summarized_scores(scores):
        fields = {
            'SupportVectorMachine_macro_f1': np.mean,
            'DecisionTree_macro_f1': np.mean,
            'RandomForest_macro_f1': np.mean,
            'NaiveBayes_macro_f1': np.mean,
            'ZeroR_macro_f1': np.mean,
        }

        return scores.groupby(['name', 'selected']).agg(fields).reset_index()

    @staticmethod
    def summarized_score_all(results_path, datasets, return_complete=False):
        complete_scoring = ResultsScorer.score_all(datasets, results_path)
        summarized_scoring = ResultsScorer._summarized_scores(complete_scoring)
        if return_complete:
            return summarized_scoring, complete_scoring
        else:
            return summarized_scoring

    @staticmethod
    def _dataset_from_result(datasets, result):
        dataset_name = result['dataset_name']
        X, y, _ = datasets.get_dataset(dataset_name).get()
        return X, y

    @staticmethod
    def _get_result_model(result):
        return {
            'name': result['name'],
            'processing_time': result['processing_time'],
            'dataset': result['dataset_name'],
            'features': result['num_features'],
            'selected': result['num_selected'],
            'sampling': result['sampling'],
            'values': result['values'],
        }

    @staticmethod
    def _print(result, verbose = DEFAULT_VERBOSE):
        if verbose > 1:
            print(
                f"{YELLOW_COLOR}Evaluated results for {GREEN_COLOR}{result['name']}\n"
                f"{WHITE_COLOR}  dataset:{CYAN_COLOR} {result['dataset']}\n"
                f"{WHITE_COLOR}  features:{CYAN_COLOR} {result['features']}{DEFAULT_COLOR}\n"
                f"{WHITE_COLOR}  selected:{CYAN_COLOR} {result['selected']}{DEFAULT_COLOR}"
            )
        elif verbose > 0:
            print(
                f"Evaluated results for {result['name']}\n"
                f"  dataset: {result['dataset']}\n"
                f"  selected: {result['selected']}\n"
                f"  features: {result['features']}"
            )
    
    @staticmethod
    def evaluate_subsets(datasets, subset_results):
        def evaluate(result):
            selected = json.loads(result['values'])
            X, y = ResultsScorer._dataset_from_result(datasets, result)
            eval_results = SelectionScorer.eval(X[:, selected], y)

            result_model = ResultsScorer._get_result_model(result)
            ResultsScorer._print(result_model)

            return pd.Series({**result_model, **flatten_dict(eval_results)})

        return subset_results.apply(evaluate, axis=1)

    @staticmethod
    def evaluate_ordered(datasets, results, evaluate_at):
        def evaluate(result):
            result_type = result['result_type']
            values = json.loads(result['values'])

            if result_type == 'rank':
                rank = values
            elif result_type == 'weights':
                rank = [int(x) for x in np.argsort(values)[::-1]]

            else:
                raise Exception("Result type must be either `rank` or `weights`")

            X, y = ResultsScorer._dataset_from_result(datasets, result)

            result_model = ResultsScorer._get_result_model(result)

            results_data = []
            for k in evaluate_at:
                if k >= len(rank):
                    continue
                selected = rank[:k]
                eval_results = SelectionScorer.eval(X[:, selected], y)
                results = {**result_model, **flatten_dict(eval_results)}
                results['selected'] = k
                results_data.append(results)
                ResultsScorer._print(results)

            return pd.DataFrame(results_data)

        return pd.concat(results.apply(evaluate, axis=1).to_list()).reset_index(drop=True)
