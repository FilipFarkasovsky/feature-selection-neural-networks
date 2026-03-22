from .loader import DatasetLoader
from .dataset import Dataset
from typing import Optional, Dict

DEFAULT_COLOR = '\033[39m'
RED_COLOR = "\033[31m"
CYAN_COLOR = '\033[36m'
GREEN_COLOR = '\033[32m'
YELLOW_COLOR = '\033[33m'

class DatasetManager:
    def __init__(self, 
            base_path: str, 
            relative_paths: Optional[Dict[str, str]] = None, 
            normalize: bool = False,
            shared: bool = True) -> None:        
        self._datasets = {}
        self.shared= shared
        self._dataloader = DatasetLoader(base_path, normalize)
        if relative_paths:
            self.add_datasets(relative_paths)

    def add_datasets(self, paths_dict: Dict[str, str]) -> None:
        for name, path in paths_dict.items():
            try:
                X, y, cols = self._dataloader.load_csv(path, to_drop=['samples'])
                self._datasets[name] = Dataset(name, X, y, cols, shared = self.shared)               
                print(f"{GREEN_COLOR}New dataset loaded successfully. Name of dataset: {name}{DEFAULT_COLOR}")
            except Exception as e:
                print(f"{RED_COLOR}Could not load {path}: {e}{DEFAULT_COLOR}")

    def get_dataset(self, name: str) -> Dataset:
        try:
            return self._datasets[name]
        except Exception:
            raise Exception(f"{RED_COLOR}There is no dataset named `{name}`.{DEFAULT_COLOR}")
