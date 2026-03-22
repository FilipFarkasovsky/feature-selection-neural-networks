import json
import os
from itertools import product, chain


from task.model import Task
from feature_selectors import feature_selectors


DEFAULT_COLOR = '\033[39m'
RED_COLOR = "\033[31m"
CYAN_COLOR = '\033[36m'
GREEN_COLOR = '\033[32m'
YELLOW_COLOR = '\033[33m'

SAMPLING_TYPES = ['none', 'bootstrap', 'percent90']


def _load_preset(path):
    file_path = os.path.join('presets', path)
    with open(file_path, "r") as f:
        return json.load(f)


def _config_to_tasks(config):
    for dataset, algorithm in product(config['datasets'], config['algorithms']):
        for params in algorithm['params']:
            for sampling in SAMPLING_TYPES:
                runs = algorithm['runs'] if sampling == 'none' else algorithm['sample_runs']
                for _ in range(runs):
                    yield Task(algorithm['name'], feature_selectors[algorithm['name']](*params), dataset, sampling)


def _print_preset(name, preset):
    print(f"Loaded configs from preset {name}:")

    for config in preset:
        description = config.get('description')
        print(f"\tDescription: {description}")
        print(f"\tDatasets:", ', '.join(config['datasets']))

def tasks_from_presets(preset_names, verbose=0):
    tasks = []
    for name in preset_names:
        try:
            _name = name if name.endswith('.json') else f'{name}.json'
            preset = _load_preset(_name)

            _print_preset(name, preset)

            for config in preset:
                tasks.append(_config_to_tasks(config))

        except Exception as e:
            print(f"Could not load preset {name} because: {e}")

    final_tasks = list(chain(*tasks))
    print(f"{YELLOW_COLOR}Generated {len(final_tasks)} tasks for {preset_names} presets!{DEFAULT_COLOR}")

    return final_tasks
