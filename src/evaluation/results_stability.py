import json
from copy import deepcopy
from multiprocessing import Pool

import numpy as np
import pandas as pd

from results.loader import ResultsLoader
from .util import rank_from_weights, keep_top_k
from .stability import stability_for_sets, stability_for_ranks, stability_for_weights


class ResultsStability:
    DEFAULT_EVALUATE_AT = [5, 10, 20, 50, 100, 200]

    @staticmethod
    def _summarize_algorithm_stability(stability):
        fields = {
            'executions': np.sum,
            'jaccard': np.mean,
            'hamming': np.mean,
            'dice': np.mean,
            'kuncheva': np.mean,
            'canberra': np.mean,
            'spearman': np.mean,
            'pearson': np.mean,
        }

        return \
            stability.drop(['dataset', 'feats'], axis=1) \
            .groupby(['name', 'selected']) \
            .agg(fields).reset_index()

    @staticmethod
    def summarized_algorithms_stability(
        results_path,
        sampling=None,
        evaluate_at = None,
        n_workers = 1,
        verbose = 1,
        evaluate_at_all_features=False,
        return_complete=False,
    ):
        evaluate_at = evaluate_at or ResultsStability.DEFAULT_EVALUATE_AT

        if sampling is not None:
            df = ResultsLoader.load_by_sampling(results_path, sampling)
        else:
            df = ResultsLoader.load_all(results_path)

        complete_stability = ResultsStability.stability_for_results(df, evaluate_at, n_workers, verbose, evaluate_at_all_features)

        summarized_stability = ResultsStability._summarize_algorithm_stability(complete_stability)
        if return_complete:
            return summarized_stability, complete_stability
        else: 
            return summarized_stability 

    @staticmethod
    def _print_evaluating(name, dataset_name, num_selected, num_executions, verbose):
        if verbose > 0:
            print(
                f"Evaluating stability for {name}:\n"
                f"  dataset: {dataset_name}\n"
                f"  number of features selected: {num_selected}\n"
                f"  executions: {num_executions}"
            )

    @staticmethod
    def _stability_for_result(df, evaluate_at, evaluate_at_all_features=True, verbose = 1):
        values = np.stack(deepcopy(df['values']).apply(json.loads).values)

        name = deepcopy(df['name'].iloc[0])
        dataset_name = deepcopy(df['dataset_name'].iloc[0])
        num_selected = deepcopy(df['num_selected'].iloc[0])
        num_features = deepcopy(df['num_features'].iloc[0])
        result_type = deepcopy(df['result_type'].iloc[0])

        num_executions = len(values)

        result_model = {
            'name': name,
            'dataset': dataset_name,
            'executions': num_executions,
            'feats': num_features,
            'selected': num_selected,
        }

        evaluate_at_k = [k for k in evaluate_at if k <= num_selected]
        if num_selected not in evaluate_at_k and evaluate_at_all_features and num_selected == num_features:
            evaluate_at_k += [num_selected]

        ranks = values
        weights = values

        if result_type == 'weights':
            ranks = np.apply_along_axis(rank_from_weights, 1, np.stack(values))

        if result_type in ['weights', 'rank']:
            all_results = []
            for k in evaluate_at_k:
                results = deepcopy(result_model)

                ResultsStability._print_evaluating(name, dataset_name, k,  num_executions, verbose)

                rank_at_k = ranks[:, :k]

                results = {
                    **results,
                    **stability_for_ranks(rank_at_k, num_features),
                    'selected': k
                }

                if result_type == 'weights':
                    if k != num_features:
                        weights_at_k = [keep_top_k(w, k, set_others_to=0) for w in weights]
                    else:
                        weights_at_k = weights

                    weights_result = stability_for_weights(weights_at_k)
                    results.update(weights_result)

                all_results.append(results)

            return pd.DataFrame(all_results)

        if result_type == 'subset':
            ResultsStability._print_evaluating(name, dataset_name, num_selected, num_executions, verbose)
            subset_results = {**result_model, **stability_for_sets(values, num_features)}
            return pd.DataFrame([subset_results])

    @staticmethod
    def stability_for_results(df,  evaluate_at, n_workers, verbose, evaluate_at_all_features=True):
        if verbose > 0:
            print("Starting stability analysis.")

        grouped = df.groupby(['name', 'dataset_name', 'num_selected'])
        groups = [g for i, g in grouped]

        if verbose > 0:
            print(f"Grouped results in {len(groups)} groups.")

        if n_workers > 1:
            with Pool(n_workers) as pool:
                args = [
                    (g, evaluate_at, evaluate_at_all_features, verbose)
                    for g in groups
                ]
                stabilities = pool.starmap(
                    ResultsStability._stability_for_result,
                    args
                )

        else:
            stabilities = []
            for g in groups:
                result = ResultsStability._stability_for_result(g, evaluate_at, evaluate_at_all_features, verbose)
                stabilities.append(result)

        return pd.concat(stabilities)
