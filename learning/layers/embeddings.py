import tensorflow_hub as hub
from keras import backend as K
from keras.engine import Layer


class ElmoEmbedding(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        self.elmo = None
        super(ElmoEmbedding, self).__init__(**kwargs)

    def build(self, input_shape):
        self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', name="{}_module".format(self.name))
        self.trainable_weights += K.tf.trainable_variables(scope="^{}_module/.*".format(self.name))
        super(ElmoEmbedding, self).build(input_shape)

    def call(self, x, mask=None):
        lengths = K.cast(K.argmax(K.cast(K.equal(x, '__PAD__'), 'uint8')), 'int32')
        result = self.elmo(inputs=dict(tokens=x, sequence_len=lengths),
                           as_dict=True,
                           signature='tokens',
                           )['elmo']
        return result

    def get_config(self):
        config = super(ElmoEmbedding, self).get_config()
        config['output_dim'] = 1024
        return config

    def compute_mask(self, inputs, mask=None):
        return K.not_equal(inputs, '__PAD__')

    def compute_output_shape(self, input_shape):
        return input_shape + (self.output_dim,)
