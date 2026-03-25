import numpy as np
from lassonet import LassoNetClassifier
from sklearn.preprocessing import LabelEncoder
from feature_selectors.base_models import BaseEmbeddedFeatureSelector
import torch

class LassoNetFeatureSelector(BaseEmbeddedFeatureSelector):
    """
    LassoNet feature selector.
    """
    DEFAULT_HIDDEN_DIMS = (32, 32, 32)

    def __init__(
        self,
        n_features=None,
        hidden_dims=None,
        **kwargs
    ):
        hidden_dims = tuple(hidden_dims) if hidden_dims else self.DEFAULT_HIDDEN_DIMS

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # --- Initialize models ---
        model = LassoNetClassifier(
            hidden_dims=hidden_dims,
            M= 10,
            path_multiplier=1.02,
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