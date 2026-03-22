from enum import Enum
from time import time
import random

import numpy as np


class SamplingType(Enum):
    NONE = 'none'
    BOOTSTRAP = 'bootstrap'
    PERCENT_90 = 'percent90'

def bootstrap(X, y=None):
    n = X.shape[0]
    idx = np.random.choice(n, n)
    return (X[idx], y[idx]) if y is not None else X[idx]

def percent90(X, y=None):
    n = X.shape[0]
    idx = np.random.choice(n, int(n * 0.9), replace=False)
    return (X[idx], y[idx]) if y is not None else X[idx]