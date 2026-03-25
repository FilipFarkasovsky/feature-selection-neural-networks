import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
import scipy
from feature_selectors.base_models.base_selector import BaseSelector, ResultType
from feature_selectors.base_models.nn_models.nn_wrapper import NNwrapper, Model
from .base_models.nn_models.deeppink import DeepPINK
from sklearn.preprocessing import StandardScaler

class Deeppink(BaseSelector):
    result_type = ResultType.WEIGHTS
    DEFAULT_HIDDEN_DIMS = (32, 32, 32)

    def __init__(self, n_features=None, hidden_dims=None, **kwargs):
        super().__init__(n_features)
        if not hidden_dims:
            print("hidden is None") 
        self.hidden_dims = tuple(hidden_dims) if hidden_dims is not None else self.DEFAULT_HIDDEN_DIMS
    
    def _create_knockoffs(self, X):
        Sigma = np.cov(X, rowvar=False)
        s = np.min(np.linalg.eigvals(Sigma)) * np.eye(X.shape[1])
        L = np.linalg.cholesky(2 * s - s @ np.linalg.inv(Sigma) @ s)
        X_knock = X - X @ np.linalg.inv(Sigma) @ s + np.random.randn(*X.shape) @ L.T
        X_aug = np.stack([X, X_knock], axis=2)
        return X_aug
    
    @staticmethod
    def generate_gaussian_knockoffs(X, eps=1e-3, lambda_=0.8):
        n_samples, n_features = X.shape

        # Compute mean and empirical 
        mu = np.mean(X, axis=0)
        sigma = np.cov(X, rowvar=False)

        # Regularize covariance for stability
        sigma_reg = lambda_ * np.diag(np.diagonal(sigma)) + (1. - lambda_) * sigma

        # Compute diagonal s matrix for knockoffs
        s = np.diagonal(sigma_reg)
        S = np.diag(s)

        # Compute matrix for knockoff covariance
        sigma_inv_S = scipy.linalg.solve(sigma_reg, S, assume_a='pos')
        V = 2. * S - np.dot(S, sigma_inv_S)
        L = np.linalg.cholesky(V + eps * np.eye(n_features))

        # Center X and Compute knockoffs
        mu_tilde = X - (X - np.broadcast_to(mu, X.shape)) @ sigma_inv_S
        X_knock = mu_tilde + np.random.normal(size=X.shape) @ L.T

        # Return original + knockoffs 
        return np.stack([X, X_knock], axis=2) 
          
    def fit(self, X, y, n_informative, **kwargs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_classes = len(set(y))

        X = StandardScaler().fit_transform(X)
        y = LabelEncoder().fit_transform(y)

        X_augmented = self.generate_gaussian_knockoffs(X)

        X_augmented = torch.tensor(X_augmented, dtype=torch.float32, device=device)
        y = torch.tensor(y, dtype=torch.long, device=device)
        
        loss_callbacks = []
        _lambda = 0.05 * np.sqrt(2.0 * np.log(X.shape[1]) / 1000)
        model = DeepPINK(Model(X.shape[1], n_classes, hidden_dims=self.hidden_dims), X.shape[1])
        model.to(device)
        for layer in model.children():
            if isinstance(layer, torch.nn.Linear):
                loss_callbacks.append(lambda l=layer: _lambda * torch.sum(torch.abs(l.weight)))
        wrapper = NNwrapper(model, n_classes)
        for loss_callback in loss_callbacks:
            wrapper.add_loss_callback(loss_callback)

        wrapper.fit(X_augmented, y, device=device)

        model.to("cpu")
        self._weights = wrapper.model.get_weights().astype(float).tolist()
        self._rank = np.argsort(self._weights)[::-1]

        if self._n_features is not None:
            self._selected = self._rank[:self._n_features]
            self._support_mask = np.zeros(X.shape[1])
            self._support_mask[self._rank] = True

        self._fitted = True
        return self
