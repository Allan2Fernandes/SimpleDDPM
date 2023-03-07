from keras import layers
import keras
from keras.layers import UpSampling2D, Conv2D
import tensorflow as tf



class UpSample(layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = (3,3)
        self.upsampling_layer = UpSampling2D(size=2, interpolation='nearest')
        self.conv_layer = Conv2D(filters=self.filters, kernel_size=self.kernel_size, padding = 'same', strides = 1, kernel_initializer=tf.keras.initializers.RandomUniform())
        pass

    def call(self,  input):
        x = self.upsampling_layer(input)
        x = self.conv_layer(x)
        return x