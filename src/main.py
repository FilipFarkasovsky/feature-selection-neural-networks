from multiprocessing import Pool, Lock
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # disables oneDNN optimizations messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=INFO, 2=WARNING, 3=ERROR
import warnings
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning

from data.dataset_manager import DatasetManager
from results.writter import ResultsWritter
from task.runner import TaskRunner
from util.shared_resources import SharedResources
from util.command_line import get_args
from task.task_factory import tasks_from_presets, get_datasets_from_presets
from itertools import chain

from evaluation.results_prediction import ResultsScorer
from evaluation.results_stability import ResultsStability
from evaluation.results_execution_time import ExecutionTimesAggregator
from data.datasets_config import datasets_relative_paths

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

    # Dataset loading and shared memory preparation
    used_datasets = get_datasets_from_presets(presets)
    filtered_paths = {
        name: path
        for name, path in datasets_relative_paths.items()
        if name in used_datasets
    }
    datasets = DatasetManager(datasets_folder_path, filtered_paths, normalize=True)

    # Feature selection stage
    if mode in ['all', 'select']:
        # FS for classical methods
        task_runner = TaskRunner(results_path, selection_filename, verbose=verbose)
        with Pool(num_workers, SharedResources.set_resources, initargs=(datasets, Lock())) as pool:
            pool.map(task_runner.run, tasks_from_presets(presets, category_filter=['classical']))

        # FS for dnn-based methods
        SharedResources.set_resources(datasets, Lock())
        tasks = tasks_from_presets(presets, category_filter=['dnn-based'])
        for task in tasks:
            task_runner.run(task)


    # Scoring of a prediction after selecting most relevant features according to FS methods
    if mode in ['all', 'scoring']:
        selection_filename = selection_filename if selection_filename.endswith('.csv') else f'{selection_filename}.csv'
        selection_results_path = os.path.join(results_path, selection_filename)
        scoring = ResultsScorer.summarized_score_all(selection_results_path, datasets,return_complete=True)
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
