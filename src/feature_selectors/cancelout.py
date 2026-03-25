import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.preprocessing import LabelEncoder

from feature_selectors.base_models import BaseEmbeddedFeatureSelector
from keras.utils import register_keras_serializable

tf.get_logger().setLevel('ERROR')

class CancelOutFeatureSelector(BaseEmbeddedFeatureSelector):
    # Fixed defaults at class level

    DEFAULT_HIDDEN_LAYERS = (32, 32, 32)

    def __init__(
        self,
        n_features=None,
        hidden_layers=None,
        activation="sigmoid",
        epochs = 400,
        cancelout_loss = True,
        lambda_1 = 0.2,
        lambda_2 = 0.1,
        batch_size = 35,
        encode_classes=False,
        **kwargs
    ):

        hidden_layers = tuple(hidden_layers) if hidden_layers else self.DEFAULT_HIDDEN_LAYERS
        self._n_features = n_features

        model = CancelOutModel(
            activation=activation,
            hidden_layers=hidden_layers,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            cancelout_loss=cancelout_loss,
            epochs=epochs,
            batch_size=batch_size,
            **kwargs
        )

        super().__init__(
            model=model,
            weights_attr="get_weights_mask",
            is_callable=True,
            n_features=n_features,
            encode_classes=encode_classes
        ) 

class CancelOutModel:

    def __init__(
        self,
        activation="sigmoid",
        lambda_1=0.2,
        lambda_2=0.1,
        cancelout_loss=True,
        epochs=100,
        batch_size=32,
        hidden_layers=None,
        verbose=0
    ):
        self.activation = activation
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.cancelout_loss = cancelout_loss
        self.epochs = epochs
        self.batch_size = batch_size
        self.hidden_layers = hidden_layers
        self.verbose = verbose


    def _build_model(self):

        inputs = keras.Input(shape=(self.input_dim,))
        
        self.cancelout_layer = CancelOutLayer(self.activation, lambda_1=self.lambda_1, lambda_2=self.lambda_2)
        x = self.cancelout_layer(inputs)
        
        for units in self.hidden_layers:
            x = keras.layers.Dense(units, activation="relu")(x)
        outputs = keras.layers.Dense(1, activation="sigmoid")(x)

        self.model = keras.Model(inputs=inputs, outputs=outputs)

        self.model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

    def fit(self, X, y, **kwargs):
        le = LabelEncoder()
        _y = le.fit_transform(y)
        self.input_dim = X.shape[1]
        self._build_model()
        self.model.fit(
            X,
            _y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            **kwargs
        )

        return self

    def get_weights_mask(self):

        w = self.cancelout_layer.get_weights()[0]
        if self.activation == "sigmoid":
            return tf.sigmoid(w).numpy()

        if self.activation == "softmax":
            return tf.nn.softmax(w).numpy()

        return w
    
@register_keras_serializable()
class CancelOutLayer(keras.layers.Layer):
    '''
    CancelOut Layer
    '''
    def __init__(self, activation='sigmoid', cancelout_loss=True, lambda_1=None, lambda_2=None, **kwargs):
        super().__init__(**kwargs)
        self.activation_name = activation  # store as string
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.cancelout_loss = cancelout_loss
        
        if activation == 'sigmoid': self.activation = tf.sigmoid
        elif activation == 'softmax': self.activation = tf.nn.softmax
        else: raise ValueError(f"Unsupported activation: {activation}")

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1],),
            initializer=tf.keras.initializers.RandomUniform(minval=-0.3, maxval=0.3),
            trainable=True)
        
    def call(self, inputs):
        activated_w = self.activation(self.w)
        self.add_loss(self.lambda_1 * tf.reduce_sum(tf.abs(activated_w)) + self.lambda_2 * tf.norm(self.w, ord=2))
        return tf.math.multiply(inputs, self.activation(self.w))
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "activation": self.activation_name,
            "cancelout_loss": self.cancelout_loss,
            "lambda_1": self.lambda_1,
            "lambda_2": self.lambda_2
        })
        return config