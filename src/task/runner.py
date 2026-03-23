import json
from time import time
import traceback

from data.sampling import bootstrap, percent90
from results.writter import ResultsWritter
from results.model import Result
from feature_selectors.base_models import ResultType
from .model import Task
from util.shared_resources import SharedResources


DEFAULT_COLOR = '\033[39m'
RED_COLOR = "\033[31m"
CYAN_COLOR = '\033[36m'
GREEN_COLOR = '\033[32m'
YELLOW_COLOR = '\033[33m'


class TaskRunner():
    def __init__(self, results_path, output_file_name, verbose=1):
        self._results_path = results_path
        self._verbose = verbose
        self._output_file_name = output_file_name

    def _log(self, msg, color=DEFAULT_COLOR, level=1):
        if self._verbose >= level:
            print(f"{color}{msg}{DEFAULT_COLOR}")

    def run(self, task: Task):
        self._log(f"Starting task {task.name} for dataset {task.dataset_name}", CYAN_COLOR)
        try:
            shared_resources = SharedResources.get()
            datasets, lock = shared_resources['datasets'], shared_resources['lock']
            
            dataset = datasets.get_dataset(task.dataset_name)
            X, y = dataset.get_instances(), dataset.get_classes()

            if task.sampling == 'bootstrap': 
                X, y = bootstrap(X, y)
            elif task.sampling == 'percent90':
                X, y = percent90(X, y)

            fs = task.feature_selector

            start = time()
            fs.fit(X, y)
            time_spent = time() - start

            if fs.result_type is ResultType.WEIGHTS:
                values = list(fs.get_weights())
            elif fs.result_type is ResultType.RANK:
                values = [int(v) for v in fs.get_rank()]
            else:
                values = [int(v) for v in fs.get_selected()]

            num_selected = task.feature_selector._n_features
            num_features = dataset.get_instances_shape()[1]

            result = Result(
                name=task.name,
                processing_time=time_spent,
                dataset_name=dataset.name,
                num_features=num_features,
                num_selected=num_selected if num_selected else num_features,
                sampling=task.sampling,
                result_type=fs.result_type.value,
                values=json.dumps(values)
            )

            with lock:
                ResultsWritter.write_result(result, self._output_file_name, self._results_path)
                self._log(f"Task {task.name} done! [{time_spent:.2f}s]", GREEN_COLOR)
            
        except Exception as e:
            self._log(f"Error in task {task.name} for dataset {task.dataset_name}: {e}", RED_COLOR, level=0)
            self._log(traceback.format_exc(limit=5), RED_COLOR, level=1)