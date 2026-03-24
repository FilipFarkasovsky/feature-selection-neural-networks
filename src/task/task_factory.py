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
                    yield Task(algorithm['name'], feature_selectors[algorithm['name']](*params), dataset, config['n_informative'], sampling)


def _print_preset(name, preset):
    print(f"Loaded configs from preset {name}:")

    for config in preset:
        description = config.get('description')
        print(f"\tDescription: {description}")
        print(f"\tDatasets:", ', '.join(config['datasets']))

def tasks_from_presets(preset_names, category_filter=None):
    tasks = []
    for name in preset_names:
        try:
            _name = name if name.endswith('.json') else f'{name}.json'
            preset = _load_preset(_name)
            _print_preset(name, preset)

            for config in preset:
                config_category = config.get("category")
                if category_filter and config_category in category_filter:
                    tasks.append(_config_to_tasks(config))

        except Exception as e:
            print(f"Could not load preset {name} because: {e}")

    final_tasks = list(chain(*tasks))
    print(f"{YELLOW_COLOR}Generated {len(final_tasks)} tasks for {preset_names} presets for methods of type:{category_filter}!{DEFAULT_COLOR}")

    return final_tasks

def get_datasets_from_presets(preset_names):
    dataset_names = set()

    for name in preset_names:
        _name = name if name.endswith('.json') else f'{name}.json'
        preset = _load_preset(_name)

        for config in preset:
            for ds in config['datasets']:
                dataset_names.add(ds)

    return dataset_names