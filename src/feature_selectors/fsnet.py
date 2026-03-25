import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch

from feature_selectors.base_models.base_selector import BaseSelector, ResultType

from .base_models.nn_models.nn_wrapper import Model
from .base_models.nn_models.fsnet import FSNet

class FSNetFeatureSelector(BaseSelector):
    """
    FSNet feature selector using a differentiable feature selection layer
    with reconstruction regularization.
    """
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
        
    def fit(self, X, y, n_informative, **kwargs):
        n_classes = len(set(y))
        n_features = X.shape[1]

        # --- Prepare FSNet model ---
        base_model = Model(2 * n_informative, n_classes, hidden_dims=self.hidden_dims)
        fsnet = FSNet(
            base_model, 
            n_features, 
            n_bins = 30,
            n_selected = 2 * n_informative, 
            n_classes = n_classes)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        fsnet.to(device)

        # --- Preprocess data ---
        X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
        y_tensor = torch.tensor(LabelEncoder().fit_transform(y), dtype=torch.long, device=device)

        # --- Train FSNet ---
        fsnet.fit(X_tensor, y_tensor)

        # --- Extract features from model ----
        self._weights = fsnet.get_feature_importances().astype(float).tolist()
        self._rank = np.argsort(self._weights)[::-1]

        if self._n_features is not None:
            self._selected = self._rank[:self._n_features]
            self._support_mask = np.zeros(X.shape[1])
            self._support_mask[self._rank] = True

        self._fitted = True
        return self
