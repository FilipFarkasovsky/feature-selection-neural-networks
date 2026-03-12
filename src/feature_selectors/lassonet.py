import numpy as np
from lassonet import LassoNetClassifier

from feature_selectors.base_models import BaseEmbeddedFeatureSelector


class LassoNetFeatureSelector(BaseEmbeddedFeatureSelector):
    def __init__(
        self,
        hidden_dims=(128, 64, 32),
        M=10,
        path_multiplier=1.02,
        n_features=None,
        **kwargs
    ):
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