from keras import layers
import keras
from PositionalEmbedding import PositionalEmbedding
import tensorflow as tf
from ConvBlock import ConvBlock
from Downsample import Downsample
from Upsample import UpSample
from AttentionBlock import AttentionBlock
from keras.layers import Concatenate, BatchNormalization, Conv2D, Dense
from keras.activations import swish
from keras import Model



def diff_model_structure(img_size, img_channels, filters):
    kernel_size = (3, 3)
    strides = (1, 1)
    image_input = layers.Input(shape=(img_size, img_size, img_channels))
    time_input = layers.Input(shape=(), dtype=tf.int32)

    # Unet architecture

    x = layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same', strides=strides, kernel_initializer=tf.keras.initializers.RandomUniform())(image_input)
    t_embedding = PositionalEmbedding(dim=filters * 4)(time_input)
    t_embedding = Dense(units=filters * 4, kernel_initializer=tf.keras.initializers.RandomUniform(), activation='swish')(t_embedding)
    t_embedding = Dense(units=filters*4, kernel_initializer=tf.keras.initializers.RandomUniform())(t_embedding)


    # Save the skip connections

    # Downsampling blocks
    x = ConvBlock(filters=filters)([x, t_embedding])
    skip1 = x
    x = Downsample(filters=filters)(x)

    x = ConvBlock(filters=filters * 2)([x, t_embedding])
    skip2 = x
    x = Downsample(filters=filters * 2)(x)

    x = ConvBlock(filters=filters * 4)([x, t_embedding])
    x = AttentionBlock(units=filters * 4)(x)
    skip3 = x
    x = Downsample(filters=filters * 4)(x)

    x = ConvBlock(filters=filters * 8)([x, t_embedding])
    x = AttentionBlock(units=filters * 8)(x)
    skip4 = x
    x = Downsample(filters=filters * 8)(x)

    # Bottleneck layer
    x = ConvBlock(filters=filters * 16)([x, t_embedding])
    x = AttentionBlock(units=filters * 16)(x)
    x = ConvBlock(filters=filters * 16)([x, t_embedding])

    x = UpSample(filters=filters * 8)(x)
    # Concatenate skips and upsampled layers
    x = Concatenate(axis=-1)([skip4, x])
    x = ConvBlock(filters=filters * 8)([x, t_embedding])
    x = AttentionBlock(units=filters * 8)(x)

    x = UpSample(filters=filters * 4)(x)
    x = Concatenate(axis=-1)([skip3, x])
    x = ConvBlock(filters=filters * 4)([x, t_embedding])
    x = AttentionBlock(units=filters * 4)(x)

    x = UpSample(filters=filters * 2)(x)
    x = Concatenate(axis=-1)([skip2, x])
    x = ConvBlock(filters=filters * 2)([x, t_embedding])

    x = UpSample(filters=filters)(x)
    x = Concatenate(axis=-1)([skip1, x])
    x = ConvBlock(filters=filters)([x, t_embedding])

    x = BatchNormalization()(x)
    x = swish(x)
    x = Conv2D(filters=img_channels, kernel_size=(3, 3), padding='same', kernel_initializer=tf.keras.initializers.RandomUniform(),
               strides=(1, 1))(x)

    model = Model(inputs=[image_input, time_input], outputs=x)
    return model