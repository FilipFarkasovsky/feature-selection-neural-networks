import json
import numpy as np
import pandas as pd

from util.dict import flatten_dict
from results.loader import ResultsLoader

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import minmax_scale

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier

from sklearn.metrics import make_scorer, accuracy_score, recall_score, precision_score, f1_score
from .measures import one_vs_all_roc_auc


class SelectionScorer:
    default_models = {
        'SupportVectorMachine': SVC(),
        'DecisionTree': DecisionTreeClassifier(),
        'RandomForest': RandomForestClassifier(),
        'NaiveBayes': GaussianNB(),
        'ZeroR': DummyClassifier()
    }

    default_scoring = {
        "accuracy": make_scorer(accuracy_score),
        "macro_f1": make_scorer(f1_score, average='macro'),
        "roc_auc": make_scorer(one_vs_all_roc_auc)
    }


    full_scoring = {
        "accuracy": make_scorer(accuracy_score),
        "macro_recall": make_scorer(recall_score, average='macro'),
        "macro_precision": make_scorer(precision_score, average='macro', zero_division=False),
        "micro_f1": make_scorer(f1_score, average='micro'),
        "macro_f1": make_scorer(f1_score, average='macro'),
        "roc_auc": make_scorer(one_vs_all_roc_auc)
    }

    @staticmethod
    def _eval(X, y, model, scoring):
        X = minmax_scale(X)
        cv = StratifiedKFold()
        results = cross_validate(model, X, y, cv=cv, scoring=scoring)
        return {k.replace("test_", ""): v.mean() for k, v in results.items() if not k.endswith("time")}

    
    @staticmethod
    def eval(X, y, models=None, scoring=None):
        models = models or SelectionScorer.default_models
        scoring = scoring or SelectionScorer.default_scoring
        return {name: SelectionScorer._eval(X, y, model, scoring) for name, model in models.items()}


class ResultsScorer:
    DEFAULT_EVALUATE_AT = [5, 10, 20, 50, 100, 200]

    @staticmethod
    def summarized_score_all(results_path, datasets, return_complete=False, evaluate_at = None):
        evaluate_at = evaluate_at or ResultsScorer.DEFAULT_EVALUATE_AT

        all_scores = []

        for rtype in ['subset', 'rank', 'weights']:
            try:
                results = ResultsLoader.load_by_result_type(results_path, rtype)
                scorer = ResultsScorer.evaluate_subsets if rtype == 'subset' else ResultsScorer.evaluate_ordered
                all_scores.append(scorer(datasets, results, evaluate_at) if rtype != 'subset' else scorer(datasets, results))
            except Exception as e:
                print(f"Could not load {rtype} results, reason: {e}")

        if not all_scores:
            raise RuntimeError("Could not generate any scoring for given results.")
        return pd.concat(all_scores, ignore_index=True)


    @staticmethod
    def _dataset_from_result(datasets, result):
        X, y, _ = datasets.get_dataset(result['dataset_name']).get()
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
    def _print(result):
        print(f"Evaluated results for {result['name']}; dataset: {result['dataset']}\n"
            f"selected: {result['selected']}, features: {result['features']}")

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