from keras import layers
import tensorflow as tf

class Downsample(layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = (3,3)
        self.downsample_layer = layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size, padding = 'same', strides = 2, kernel_initializer=tf.keras.initializers.RandomUniform())
        pass

    def call(self,  input):
        x = self.downsample_layer(input)
        return x
    pass