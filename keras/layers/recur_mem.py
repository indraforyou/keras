from keras import backend as K
from keras.engine.topology import Layer

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
        # self.input_shape = input_shape
        self.out_shape = (self.time_dim,)+input_shape[1:]
        self.W = K.zeros(self.out_shape)
        # self.W = []
        # for _ in range(self.time_dim):
        #     # self.W.append(self.add_weight(shape=(input_shape[1], self.time_dim),
        #     #                      initializer='random_uniform',
        #     #                      trainable=False))
        #     self.W.append(K.zeros((1,)+input_shape[1:]))
        # # super(RecurDataMem, self).build()
        # # super(Dense, self).__init__(**kwargs)

    def call(self, x, mask=None):
        # for idx in range(self.time_dim-1):
        #     self.W[idx] = self.W[idx+1]
        # # return K.dot(x, self.W)
        # return K.concatenate(self.W[1:]+[x], axis=0)
        out_list = K.tf.unpack(self.W)
        out = K.stack(out_list[1:]+[x[0]])
        self.add_update([(self.W, out)])
        return out

    def get_output_shape_for(self, input_shape):
        return self.out_shape


