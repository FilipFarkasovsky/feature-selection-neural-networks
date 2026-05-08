import os
import numpy as np
from itertools import chain

DATASETS_PATH = "datasets/friedman"
np.random.seed(1)


class DatasetParams:
    def __init__(self, n_samples: int, n_features: int, noise_std: float = 0.5):
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_informative = 6  # x1..x
        self.n_noisy = n_features - self.n_informative
        self.noise_std = noise_std

    def name(self):
        return f"friedman1_{self.n_samples}samples_{self.n_features}features"

    def csv_name(self):
        return self.name() + ".csv"

    def feature_names(self):
        types = chain(
            ("informative" for _ in range(self.n_informative)),
            ("noisy" for _ in range(self.n_noisy))
        )
        return [f"{t}_{i}" for i, t in enumerate(types)]

    def build_dataset(self):
        # All features uniform [0,1]
        X_inf = np.random.uniform(0, 1, size=(self.n_samples, self.n_informative))

        f = (
            20 * (X_inf[:, 0] - 0.5) * (X_inf[:, 1] - 0.5) # XOR Continous
            + 20 * (X_inf[:, 2] - 0.5) ** 2             # Quadratic
            + 15 * (X_inf[:, 3] * X_inf[:, 4])          # Multiplicative
            + 10 * X_inf[:, 5]                          # Linear Control
            + np.random.normal(0, self.noise_std, self.n_samples) # Noise
        )

        # Center f(x)
        f_centered = f - np.median(f)

        # Logistic transformation
        prob = 1 / (1 + np.exp(-f_centered))

        # Sample labels\
        y = np.random.binomial(1, prob)

        # Add noisy features
        if self.n_noisy > 0:
            X_noise = np.random.uniform(0, 1, size=(self.n_samples, self.n_noisy))
            X = np.concatenate((X_inf, X_noise), axis=1)
        else:
            X = X_inf

        return X, y, self.feature_names()

    def to_csv(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        X, y, cols = self.build_dataset()
        data = np.column_stack((X, y))

        # all continuous features
        specs = ["%.6f" for _ in cols] + ["%.6f"]
        cols += ["class"]

        header = ",".join(f'"{c}"' for c in cols)
        path_to_save = os.path.join(path, self.csv_name())

        np.savetxt(
            path_to_save,
            data,
            fmt=specs,
            header=header,
            delimiter=",",
            comments="",
        )

        print(f"Saved dataset to {path_to_save}")


# Example usage
dataset = DatasetParams(n_samples=1000, n_features=8, noise_std=1.0)
dataset.to_csv(DATASETS_PATH)

dataset = DatasetParams(n_samples=1000, n_features=16, noise_std=1.0)
dataset.to_csv(DATASETS_PATH)

dataset = DatasetParams(n_samples=1000, n_features=32, noise_std=1.0)
dataset.to_csv(DATASETS_PATH)

dataset = DatasetParams(n_samples=1000, n_features=64, noise_std=1.0)
dataset.to_csv(DATASETS_PATH)

dataset = DatasetParams(n_samples=1000, n_features=128, noise_std=1.0)
dataset.to_csv(DATASETS_PATH)

dataset = DatasetParams(n_samples=1000, n_features=256, noise_std=1.0)
dataset.to_csv(DATASETS_PATH)

dataset = DatasetParams(n_samples=1000, n_features=512, noise_std=1.0)
dataset.to_csv(DATASETS_PATH)

dataset = DatasetParams(n_samples=1000, n_features=768, noise_std=1.0)
dataset.to_csv(DATASETS_PATH)

dataset = DatasetParams(n_samples=1000, n_features=1024, noise_std=1.0)
dataset.to_csv(DATASETS_PATH)

dataset = DatasetParams(n_samples=1000, n_features=896, noise_std=1.0)
dataset.to_csv(DATASETS_PATH)

dataset = DatasetParams(n_samples=1000, n_features=2048, noise_std=1.0)
dataset.to_csv(DATASETS_PATH)
