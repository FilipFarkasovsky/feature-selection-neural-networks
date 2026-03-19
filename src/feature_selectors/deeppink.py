import numpy as np
from sklearn.preprocessing import label_binarize

from feature_selectors.base_models.base_selector import BaseSelector, ResultType
from feature_selectors.base_models.nn_models.nn_wrapper import NNwrapper

from sklearn.preprocessing import StandardScaler

class Deeppink(BaseSelector):
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

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Estimate covariance
        Sigma = np.cov(X, rowvar=False)
        # Choose s (diagonal matrix) smaller than smallest eigenvalue of Sigma
        eigvals = np.linalg.eigvals(Sigma)
        s = np.min(eigvals) * np.eye(X.shape[1])
        # Cholesky decomposition
        L = np.linalg.cholesky(2*s - s @ np.linalg.inv(Sigma) @ s)
        # Generate knockoffs
        X_knock = X - X @ np.linalg.inv(Sigma) @ s + np.random.randn(*X.shape) @ L.T

        X_augmented = np.empty((X.shape[0], X.shape[1], 2))
        X_augmented[:, :, 0] = X
        X_augmented[:, :, 1] = X_knock

        wrapper = NNwrapper.create(X.shape[1], n_classes, arch='deeppink', hidden_dims=self.hidden_dims)

        wrapper.fit(X_augmented, y)
        self._weights = wrapper.model.get_weights().astype(float).tolist()
        self._rank = np.argsort(self._weights)[::-1]

        if self._n_features is not None:
            self._selected = self._rank[:self._n_features]
            self._support_mask = np.zeros(X.shape[1])
            self._support_mask[self._rank] = True

        self._fitted = True
        return self
