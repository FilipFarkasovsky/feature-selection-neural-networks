from .dataloader import DataLoader
from .shared_dataset import SharedDataset


class SharedDatasets:
    def __init__(self, base_path, relative_paths,normalize=False):
        self._datasets = {}
        self._dataloader = DataLoader(base_path, normalize)
        if relative_paths:
            self.add_datasets(relative_paths)



    def add_datasets(self, paths_dict):
        for name, path in paths_dict.items():
            try:
                X, y, cols = self._dataloader.load(path, to_drop=['samples'])
                sd = SharedDataset(name, X, y, cols)
                self._datasets[name] = sd                
                print(f"Added {name} dataset successfully")
            except Exception as e:
                print(f"Could not load {path}: {e}")

    def get_dataset(self, name):
        try:
            return self._datasets[name]
        except Exception:
            raise Exception(f"There is no dataset named `{name}`.")
