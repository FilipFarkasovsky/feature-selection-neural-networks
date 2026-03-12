from itertools import chain
import os

import numpy as np


DATASETS_PATH = "datasets/xor3"
np.random.seed(1)


class DatasetParams:
    def __init__(self, n_samples: int, n_features: int):
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_informative = 3
        self.n_noisy = n_features - self.n_informative

    def name(self):
        return f"xor3_{self.n_samples}samples_{self.n_features}features"

    def csv_name(self):
        return self.name() + ".csv"

    def feature_names(self):
        types = chain(
            ("informative" for _ in range(self.n_informative)),
            ("noisy" for _ in range(self.n_noisy))
        )
        return [f"{t}_{i}" for i, t in enumerate(types)]

    def build_dataset(self):
        # informative continuous variables
        X_inf = np.random.normal(size=(self.n_samples, self.n_informative))

        # high-order XOR rule
        y = (
            (X_inf[:, 0] > 0)
            ^ (X_inf[:, 1] > 0)
            ^ (X_inf[:, 2] > 0)
        ).astype(int)

        # noisy continuous features
        X_noise = np.random.normal(size=(self.n_samples, self.n_noisy))

        X = np.concatenate((X_inf, X_noise), axis=1)

        return X, y, self.feature_names()

    def to_csv(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        X, y, cols = self.build_dataset()
        data = np.column_stack((X, y))

        specs = ['%.6f' for _ in cols] + ['%i']
        cols += ['class']

        header = ','.join(f'"{c}"' for c in cols)
        path_to_save = os.path.join(path, self.csv_name())

        np.savetxt(
            path_to_save,
            data,
            fmt=specs,
            header=header,
            delimiter=',',
            comments=''
        )

        print(f"Saved dataset to {path_to_save}")


dataset = DatasetParams(n_samples=500, n_features=8)
dataset.to_csv(DATASETS_PATH)

dataset = DatasetParams(n_samples=500, n_features=16)
dataset.to_csv(DATASETS_PATH)

dataset = DatasetParams(n_samples=500, n_features=32)
dataset.to_csv(DATASETS_PATH)

dataset = DatasetParams(n_samples=500, n_features=64)
dataset.to_csv(DATASETS_PATH)

dataset = DatasetParams(n_samples=500, n_features=128)
dataset.to_csv(DATASETS_PATH)

dataset = DatasetParams(n_samples=500, n_features=256)
dataset.to_csv(DATASETS_PATH)

dataset = DatasetParams(n_samples=500, n_features=512)
dataset.to_csv(DATASETS_PATH)

dataset = DatasetParams(n_samples=500, n_features=1024)
dataset.to_csv(DATASETS_PATH)