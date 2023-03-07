from keras import layers
import tensorflow as tf

class AttentionBlock(layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.b_norm = layers.BatchNormalization()
        self.query = layers.Dense(units, kernel_initializer=tf.keras.initializers.RandomUniform())
        self.key = layers.Dense(units, kernel_initializer=tf.keras.initializers.RandomUniform())
        self.value = layers.Dense(units, kernel_initializer=tf.keras.initializers.RandomUniform())
        self.process_output = layers.Dense(units, kernel_initializer=tf.keras.initializers.RandomUniform())

    def call(self, inputs):
        inputs = self.b_norm(inputs)
        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)

        _, _, _, dim = k.shape
        product = tf.einsum("b h i d, b h j d -> b h i j", q, k)

        scaled_product = product*dim**(-0.5)
        attention_score = tf.keras.activations.softmax(scaled_product)
        output = tf.einsum('b h i j, b h j d -> b h i d', attention_score, v)
        output = self.process_output(output)
        return inputs + output
