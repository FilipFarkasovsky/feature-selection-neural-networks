import numpy as np
from sklearn.preprocessing import LabelEncoder

from feature_selectors.base_models.base_selector import BaseSelector, ResultType

from .base_models.nn_models.nn_wrapper import Model
from .base_models.nn_models.fsnet import FSNet

class FSNetFeatureSelector(BaseSelector):
    result_type = ResultType.WEIGHTS
    DEFAULT_HIDDEN_DIMS = (32, 32, 32)

    def __init__(
        self,
        n_features=None,
        hidden_dims=None,
        **kwargs
    ):
        super().__init__(n_features)
        self.hidden_dims = tuple(hidden_dims) if hidden_dims is not None else self.DEFAULT_HIDDEN_DIMS
        
        if not isinstance(hidden_dims, tuple):
            self.hidden_dims = tuple(hidden_dims)
        
    def fit(self, X, y, **kwargs):
        n_classes = len(set(y))

        n_selected = min(self._n_features, X.shape[1])
        fsnet = FSNet(Model(n_selected, n_classes, hidden_dims=self.hidden_dims), X.shape[1], 30, n_selected, n_classes)
        le = LabelEncoder()
        y = le.fit_transform(y)
        fsnet.fit(X, y)
        self._weights = fsnet.get_feature_importances().astype(float).tolist()
        self._rank = np.argsort(self._weights)[::-1]

        if self._n_features is not None:
            self._selected = self._rank[:self._n_features]
            self._support_mask = np.zeros(X.shape[1])
            self._support_mask[self._rank] = True

        self._fitted = True
        return self
