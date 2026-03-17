from multiprocessing import Pool, Lock
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # disables oneDNN optimizations messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=INFO, 2=WARNING, 3=ERROR
import warnings
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning

from data.shared_datasets import SharedDatasets
from results.writter import ResultsWritter
from task.runner import TaskRunner
from util.shared_resources import SharedResources
from util.command_line import get_args
from util.task_creation_helper import tasks_from_presets

from evaluation.results_scorer import ResultsScorer
from evaluation.results_stability import ResultsStability
from evaluation.results_execution_time import ExecutionTimesAggregator

from multiprocessing import cpu_count
from time import time
warnings.filterwarnings(
    "ignore",
    message="TensorFlow GPU support is not available on native Windows.*",
    module="tensorflow.*"
)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow.*")
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=ConvergenceWarning)
warnings.simplefilter(action='ignore', category=UndefinedMetricWarning)
ConvergenceWarning('ignore')
UndefinedMetricWarning('ignore')


def main():
    args = get_args()
    current_timestamp = int(time())

    # configurations
    mode = args.mode
    verbose = args.verbose
    presets = args.presets
    presets_runs = args.presets_runs
    num_workers = args.workers

    # paths and file names configs
    results_path = args.results_path
    datasets_folder_path = args.datasets_path
    selection_filename = args.selection_filename
    scoring_filename = args.scoring_filename
    determinism_filename = args.determinism_filename
    stability_filename = args.stability_filename
    times_filename = args.times_filename

    datasets_relative_paths = {
            # Xor Dataset
            'xor_500samples_50features': 'xor/xor_500samples_50features.csv',
            'friedman1_1000samples_128features': 'friedman/friedman1_1000samples_128features.csv',

            # Cumida Datasets
            'Liver_GSE22405': 'cumida/Liver_GSE22405.csv',
            'Prostate_GSE6919_U95C': 'cumida/Prostate_GSE6919_U95C.csv',

            # Synthetic Datasets
            'synth_100samples_5000features_50informative':
                'synthetic/synth_100samples_5000features_50informative.csv',
            'synth_100samples_5000features_50informative_50redundant':
                'synthetic/synth_100samples_5000features_50informative_50redundant.csv',
        }

    # Dataset loading and shared memory preparation
    datasets = SharedDatasets(datasets_folder_path, datasets_relative_paths, normalize=True)

    # Feature selection stage
    if mode in ['all', 'select']:
        task_runner = TaskRunner(results_path, selection_filename, verbose=verbose)
        with Pool(num_workers, SharedResources.set_resources, initargs=(datasets, Lock())) as pool:
            pool.map(task_runner.run, tasks_from_presets(presets, presets_runs, verbose=verbose))


    # Scoring of feature selection results
    if mode in ['all', 'scoring']:
        selection_filename = selection_filename if selection_filename.endswith('.csv') else f'{selection_filename}.csv'
        selection_results_path = os.path.join(results_path, selection_filename)
        summarized_scoring, scoring = ResultsScorer.summarized_score_all(selection_results_path, datasets,return_complete=True)

        ResultsWritter.write_dataframe(summarized_scoring, scoring_filename, results_path)
        ResultsWritter.write_dataframe(scoring, f'{scoring_filename}-complete', results_path)

    # Stability analysis of feature selection methods
    if mode in ['stability']:
        try:
            alg_stab_sum, alg_stab = ResultsStability.summarized_algorithms_stability(
                selection_results_path, sampling='bootstrap', return_complete=True, verbose = verbose, n_workers = num_workers
            )

            ResultsWritter.write_dataframe(alg_stab_sum, f'{stability_filename}-bootstrap', results_path)
            ResultsWritter.write_dataframe(alg_stab, f'{stability_filename}-bootstrap-complete', results_path)
        except Exception as e:
            print(f"Could not run data stability evaluation. Reason: {e}")

        try:
            alg_stab_sum, alg_stab = ResultsStability.summarized_algorithms_stability(
                selection_results_path, sampling='percent90', return_complete=True, verbose = verbose, n_workers = num_workers
            )

            ResultsWritter.write_dataframe(alg_stab_sum, f'{stability_filename}-90perecent', results_path)
            ResultsWritter.write_dataframe(alg_stab, f'{stability_filename}-90percent-complete', results_path)
        except Exception as e:
            print(f"Could not run data stability evaluation. Reason: {e}")

    if mode in ['determinism']:
        try:
            alg_det_sum, alg_det = ResultsStability.summarized_algorithms_stability(
                selection_results_path, sampling='none', return_complete=True, verbose = verbose, n_workers = num_workers
            )

            ResultsWritter.write_dataframe(alg_det_sum, determinism_filename, results_path)
            ResultsWritter.write_dataframe(alg_det, f'{determinism_filename}-complete', results_path)
        except Exception as e:
            print(f"Could not run results determinism evaluation. Reason: {e}")

    # execution times evaluation
    if mode in ['all', 'times']:
        exec_times = ExecutionTimesAggregator.aggregated_execution_times(selection_results_path)
        ResultsWritter.write_dataframe(exec_times, times_filename, results_path)


if __name__ == '__main__':
    main()
