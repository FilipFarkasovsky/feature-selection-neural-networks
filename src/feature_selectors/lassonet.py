import numpy as np
from lassonet import LassoNetClassifier
from sklearn.preprocessing import LabelEncoder
from feature_selectors.base_models import BaseEmbeddedFeatureSelector
import torch

class LassoNetFeatureSelector(BaseEmbeddedFeatureSelector):
    DEFAULT_HIDDEN_DIMS = (32, 32, 32)
    DEFAULT_M = 10
    DEFAULT_PATH_MULTIPLIER = 1.02

    def __init__(
        self,
        n_features=None,
        hidden_dims=None,
        path_multiplier=None,
        M=None,
        **kwargs
    ):
        hidden_dims = tuple(hidden_dims) if hidden_dims is not None else self.DEFAULT_HIDDEN_DIMS
        M = M if M is not None else self.DEFAULT_M
        path_multiplier = path_multiplier if path_multiplier is not None else self.DEFAULT_PATH_MULTIPLIER

        if not isinstance(hidden_dims, tuple):
            hidden_dims = tuple(hidden_dims)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = LassoNetClassifier(
            hidden_dims=hidden_dims,
            M=M,
            path_multiplier=path_multiplier,
            verbose = False,
            n_iters = (300,100),
            device=device,
            **kwargs
        )

        super().__init__(
            model=model,
            weights_attr="feature_importances_",
            is_callable=False,
            n_features=n_features,
            encode_classes=False
        )

    def fit(self, X, y, n_informative):
        le = LabelEncoder()
        y = le.fit_transform(y)
        super().fit(X, y, n_informative=n_informative)