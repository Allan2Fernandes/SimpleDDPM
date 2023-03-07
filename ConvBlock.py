from keras import layers
import tensorflow as tf
from keras.layers import Dense, Conv2D, Add, BatchNormalization
from keras.activations import swish

class ConvBlock(layers.Layer):
    def __init__(self, filters, training=True, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = (3, 3)
        self.strides = (1, 1)

        self.embed_resize_layer = Dense(self.filters, kernel_initializer=tf.keras.initializers.RandomUniform())
        self.res_conv = Conv2D(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides, padding='same',kernel_initializer=tf.keras.initializers.RandomUniform())

        self.conv1 = Conv2D(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides, padding='same',kernel_initializer=tf.keras.initializers.RandomUniform())
        self.conv2 = Conv2D(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides, padding='same',kernel_initializer=tf.keras.initializers.RandomUniform())
        self.bnorm1 = BatchNormalization()
        self.bnorm2 = BatchNormalization()
        self.Add1 = Add()
        self.Add2 = Add()

        pass

    def call(self, inputs):
        x, t_embedding = inputs
        t_embedding = self.embed_resize_layer(t_embedding)
        t_embedding = tf.expand_dims(t_embedding, axis=1)
        t_embedding = tf.expand_dims(t_embedding, axis=1)
        # Conv(c) and save for res
        residual_connection = self.res_conv(x)
        # BN
        x = self.bnorm1(x)
        # A
        x = swish(x)
        # Conv(c)
        x = self.conv1(x)
        # ADD t
        x = self.Add1([x, t_embedding])
        # BN
        x = self.bnorm2(x)
        # A
        x = swish(x)
        # Conv(c)
        x = self.conv2(x)
        # ADD residual
        x = self.Add2([x, residual_connection])
        return x