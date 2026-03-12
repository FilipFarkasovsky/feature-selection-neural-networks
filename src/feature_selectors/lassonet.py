import numpy as np
from lassonet import LassoNetClassifier

from feature_selectors.base_models import BaseEmbeddedFeatureSelector


class LassoNetFeatureSelector(BaseEmbeddedFeatureSelector):
    def __init__(
        self,
        hidden_dims=None,
        M=None,
        path_multiplier=None,
        n_features=None,
        **kwargs
    ):
        if hidden_dims is None:
            hidden_dims = (128, 64, 32)
        if M is None:
            M = 10
        if path_multiplier is None:
            path_multiplier = 1.02

        if not isinstance(hidden_dims, tuple):
            hidden_dims = tuple(hidden_dims)

        model = LassoNetClassifier(
            hidden_dims=hidden_dims,
            M=M,
            path_multiplier=path_multiplier,
            verbose = False,
            **kwargs
        )

        super().__init__(
            model=model,
            weights_attr="feature_importances_",
            is_callable=False,
            n_features=n_features,
            encode_classes=False
        )