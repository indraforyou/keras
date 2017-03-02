from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

class RecurDataMem(Layer):
    def __init__(self,
                    time_dim,
                    input_dim=None,
                    **kwargs):
        self.time_dim = time_dim
        self.input_dim = input_dim

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)

        super(RecurDataMem, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        print input_shape
        self.W = []
        for _ in range(self.time_dim):
            self.W.append(self.add_weight(shape=(input_shape[1], self.time_dim),
                                 initializer='random_uniform',
                                 trainable=False))
        super(RecurDataMem, self).build()

    def call(self, x):
        return K.dot(x, self.W)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)


