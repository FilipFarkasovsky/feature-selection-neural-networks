import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
from feature_selectors.base_models.base_selector import BaseSelector, ResultType
from feature_selectors.base_models.nn_models.nn_wrapper import NNwrapper, Model
from .base_models.nn_models.deeppink import DeepPINK
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

        
        loss_callbacks = []
        _lambda = 0.05 * np.sqrt(2.0 * np.log(X.shape[1]) / 1000)
        model = DeepPINK(Model(X.shape[1], n_classes, hidden_dims=self.hidden_dims), X.shape[1])
        for layer in model.children():
            # if isinstance(layer, torch.nn.Linear) and (layer.out_features > 1):
            if isinstance(layer, torch.nn.Linear):
                loss_callbacks.append(lambda l=layer: _lambda * torch.sum(torch.abs(l.weight)))
        wrapper = NNwrapper(model, n_classes)
        for loss_callback in loss_callbacks:
            wrapper.add_loss_callback(loss_callback)

        le = LabelEncoder()
        y = le.fit_transform(y)
        wrapper.fit(X_augmented, y)
        self._weights = wrapper.model.get_weights().astype(float).tolist()
        self._rank = np.argsort(self._weights)[::-1]

        if self._n_features is not None:
            self._selected = self._rank[:self._n_features]
            self._support_mask = np.zeros(X.shape[1])
            self._support_mask[self._rank] = True

        self._fitted = True
        return self
