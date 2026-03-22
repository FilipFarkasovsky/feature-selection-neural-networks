import os
from typing import List

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from sklearn.preprocessing import minmax_scale


class DatasetLoader:
    def __init__(self, base_path: str, normalize:bool=False):
        self._base_path: str = base_path
        self._normalize: bool = normalize

    def load_csv(
        self,
        path: str,
        targets: List[str] = ['type', 'class', 'target'],
        to_drop: List[str] = []
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        full_path = os.path.join(self._base_path, path)
        if not full_path.endswith('.csv'):
            raise Exception("Only .csv format currently supported.")
        
        df = pd.read_csv(full_path)
        
        targets = [x for x in targets if x in df.columns]
        to_drop = [x for x in to_drop if x in df.columns]

        X = df.drop(targets + to_drop, axis=1).values

        if self._normalize:
            X = minmax_scale(X)

        if not len(targets) > 0:
            raise Exception("No target foud")
        
        targets = targets[0] if len(targets) == 1 else targets
        y = np.array(list(df[targets].values))

        columns = np.array([x for x in df.columns if x not in targets and x not in to_drop])

        return X, y, columns