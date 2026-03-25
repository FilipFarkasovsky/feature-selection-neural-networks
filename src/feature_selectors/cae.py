import numpy as np
from sklearn.preprocessing import LabelEncoder
from feature_selectors.base_models.base_selector import BaseSelector, ResultType
from .base_models.nn_models.concrete_autoencoder import ConcreteAutoencoderFeatureSelector
import keras

class CAEFeatureSelector(BaseSelector):
    """
    Concrete Autoencoder (CAE) feature selector.
    Returns a subset of features selected via differentiable feature selection.
    """
    result_type = ResultType.SUBSET
    DEFAULT_HIDDEN_DIMS = (32, 32, 32)

    def __init__(self, n_features=None, hidden_dims=None, **kwargs):
        super().__init__(n_features)
        self.hidden_dims = tuple(hidden_dims) if hidden_dims is not None else self.DEFAULT_HIDDEN_DIMS
        
    def fit(self, X, y, n_informative, **kwargs):
        n_classes = len(np.unique(y))
        
        # --- Define the neural network for the CAE ---
        def nn(x):
            n_out = 1 if (n_classes <= 2) else n_classes
            x = keras.layers.GaussianNoise(0.1)(x)
            for dim in self.hidden_dims:
                x = keras.layers.Dense(dim)(x)
                x = keras.layers.Dropout(0.2)(x)
                x = keras.layers.ReLU(negative_slope=0.2)(x)
            x = keras.layers.Dense(n_out, activation='sigmoid')(x)
            return x

        selector = ConcreteAutoencoderFeatureSelector(
            K= 2 * n_informative,
            output_function=nn, 
            start_temp=10, 
            min_temp=0.01,
            num_epochs=400,
            learning_rate=0.0001,
            tryout_limit=1)
        
        # --- Preprocessing --- 
        y_encoded = LabelEncoder().fit_transform(y)

        # --- Model training ---
        selector.fit(X, y_encoded)

        # --- Feature selection ---
        self._selected = selector.get_support(indices=True).flatten()
        self._support_mask = np.zeros(X.shape[1])
        self._support_mask[self._selected] = True

        self._fitted = True
        return self
