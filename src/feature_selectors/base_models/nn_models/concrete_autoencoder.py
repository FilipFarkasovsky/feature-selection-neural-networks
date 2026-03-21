import math
from keras import backend as K
from keras import Model
import tensorflow as tf
from keras.layers import Layer, Softmax, Input
from keras.callbacks import EarlyStopping
from keras.initializers import Constant, glorot_normal
from keras.optimizers import Adam

class ConcreteSelect(Layer):
    
    def __init__(self, output_dim, start_temp = 10.0, min_temp = 0.1, alpha = 0.99999, **kwargs):
        self.output_dim = output_dim
        self.start_temp = start_temp
        self.min_temp = tf.constant(min_temp, dtype=tf.float32)
        self.alpha = tf.constant(alpha, dtype=tf.float32)
        super(ConcreteSelect, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.temp = self.add_weight(name = 'temp', shape = [], initializer = Constant(self.start_temp), trainable = False)
        self.logits = self.add_weight(name = 'logits', shape = [self.output_dim, input_shape[1]], initializer = glorot_normal(), trainable = True)
        super(ConcreteSelect, self).build(input_shape)
        
    def call(self, X, training = None):
        uniform = tf.random.uniform(shape=tf.shape(self.logits), minval=1e-7, maxval=1.0)
        gumbel = -tf.math.log(-tf.math.log(uniform))
        
        self.temp.assign(tf.maximum(self.min_temp, self.temp * self.alpha))
        
        noisy_logits = (self.logits + gumbel) / self.temp
        samples = tf.nn.softmax(noisy_logits)
        
        discrete_logits = tf.one_hot(tf.argmax(self.logits, axis=-1), depth=self.logits.shape[1])
        
        self.selections = tf.cond(
            tf.cast(training, tf.bool),
            lambda: samples,
            lambda: discrete_logits
        )

        Y = tf.matmul(X, tf.transpose(self.selections))
        return Y
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
    
class StopperCallback(EarlyStopping):
    
    def __init__(self, mean_max_target = 0.998):
        self.mean_max_target = mean_max_target
        super(StopperCallback, self).__init__(monitor = '', patience = float('inf'), verbose = 1, mode = 'max', baseline = self.mean_max_target)
       
    def get_monitor_value(self, logs=None):
        logits = self.model.get_layer('concrete_select').logits
        probs = tf.nn.softmax(logits, axis=-1)
        max_probs = tf.reduce_max(probs, axis=-1)
        mean_max = tf.reduce_mean(max_probs)
        return mean_max.numpy()


class ConcreteAutoencoderFeatureSelector():
    
    def __init__(self, K, output_function, num_epochs = 300, batch_size = None, learning_rate = 0.001, start_temp = 10.0, min_temp = 0.1, tryout_limit = 5):
        self.K = K
        self.output_function = output_function
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.start_temp = start_temp
        self.min_temp = min_temp
        self.tryout_limit = tryout_limit
        
    def fit(self, X, Y, val_X = None, val_Y = None):
        assert len(X) == len(Y)
        validation_data = None
        if val_X is not None and val_Y is not None:
            assert len(val_X) == len(val_Y)
            validation_data = (val_X, val_Y)
        
        if self.batch_size is None:
            self.batch_size = max(len(X) // 256, 16)
        
        num_epochs = self.num_epochs
        steps_per_epoch = (len(X) + self.batch_size - 1) // self.batch_size
        
        for i in range(self.tryout_limit):
            
            inputs = Input(shape = X.shape[1:])

            alpha = math.exp(math.log(self.min_temp / self.start_temp) / (num_epochs * steps_per_epoch))
            
            self.concrete_select = ConcreteSelect(self.K, self.start_temp, self.min_temp, alpha, name = 'concrete_select')

            selected_features = self.concrete_select(inputs)

            outputs = self.output_function(selected_features)

            self.model = Model(inputs, outputs)

            self.model.compile(Adam(self.learning_rate), loss = 'mean_squared_error')
            
            stopper_callback = StopperCallback()
            
            hist = self.model.fit(X, Y, self.batch_size, num_epochs, verbose = 0, callbacks = [stopper_callback], validation_data = validation_data)#, validation_freq = 10)
            
            mean_max_prob = tf.reduce_mean(tf.reduce_max(tf.nn.softmax(self.concrete_select.logits, axis=-1), axis=-1)).numpy()

            if mean_max_prob >= stopper_callback.mean_max_target:
                break
            
            num_epochs *= 2
        
        self.probabilities = tf.nn.softmax(self.model.get_layer('concrete_select').logits, axis=-1).numpy()
        self.indices = tf.argmax(self.model.get_layer('concrete_select').logits, axis=-1).numpy()          
        
        return self
    
    def get_indices(self):
        return tf.argmax(self.model.get_layer('concrete_select').logits, axis=-1).numpy()    
    
    def get_mask(self):
        return tf.reduce_sum(tf.one_hot(tf.argmax(self.model.get_layer('concrete_select').logits, axis=-1), depth=self.model.get_layer('concrete_select').logits.shape[1]), axis=0).numpy()    
    def transform(self, X):
        return X[self.get_indices()]
    
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
    
    def get_support(self, indices = False):
        return self.get_indices() if indices else self.get_mask()
    
    def get_params(self):
        return self.model