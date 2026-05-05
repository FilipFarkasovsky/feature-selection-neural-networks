import time
from util.dict import flatten_dict
from results.writter import ResultsWritter
import pandas as pd
from evaluation.neural_network_prediction import run_pipeline

def summary_before_prediction(used_datasets= None, datasets=None, scoring_filename=None,results_path = None ):
    results_data = []

    for name in used_datasets:
        X, y, _ = datasets.get_dataset(name).get()

        nn_start = time.time()
        nn_out = run_pipeline(X, y, verbose=False)
        nn_time = time.time() - nn_start

        results = {
            "dataset": name,
            "nn_macro_f1": nn_out["macro_f1"],
            "nn_accuracy": nn_out["accuracy"],
            "nn_time": nn_time,
            "nn_epochs": nn_out["epochs_trained"]
        }

        results = flatten_dict(results)
        results_data.append(results)

    tab = pd.DataFrame(results_data)
    tab = tab.set_index("dataset", drop=False)  # rows = datasets

    ResultsWritter.write_dataframe(tab, f'{scoring_filename}-before-pred', results_path)