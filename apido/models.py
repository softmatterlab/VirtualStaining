from . import deeptrack as dt
from tensorflow.keras import layers, backend as K

from tensorflow.keras.initializers import RandomNormal
import numpy as np


def generator(breadth, depth, config):
    """"""

    sub = np.array(config["input_sub"]).reshape((1, 1, 1, -1))
    div = np.array(config["input_div"]).reshape((1, 1, 1, -1))
    normalization_layer = layers.Lambda(
        lambda x: K.tanh(3 * (x - sub) / (div - sub) - 1.5)
    )

    kernel_initializer = RandomNormal(mean=0.0, stddev=0.02)

    activation = layers.LeakyReLU(0.2)

    convolution_block = dt.layers.ConvolutionalBlock(
        activation=activation,
        instance_norm=True,
        kernel_initializer=kernel_initializer,
    )

    base_block = dt.layers.ResidualBlock(
        activation=activation, kernel_initializer=kernel_initializer
    )

    pooling_block = dt.layers.ConvolutionalBlock(
        strides=2,
        activation=activation,
        instance_norm=True,
        kernel_initializer=kernel_initializer,
    )

    upsample_block = dt.layers.StaticUpsampleBlock(
        kernel_size=3,
        instance_norm=True,
        activation=activation,
        with_conv=False,
        kernel_initializer=kernel_initializer,
    )

    generator = dt.models.unet(
        input_shape=(None, None, 7),
        conv_layers_dimensions=list(breadth * 2 ** n for n in range(depth - 1)),
        base_conv_layers_dimensions=(breadth * 2 ** (depth - 1),),
        output_conv_layers_dimensions=(
            breadth,
            breadth // 2,
        ),
        steps_per_pooling=2,
        number_of_outputs=3,
        output_kernel_size=1,
        scale_output=True,
        input_layer=normalization_layer,
        encoder_convolution_block=convolution_block,
        decoder_convolution_block=convolution_block,
        base_convolution_block=base_block,
        pooling_block=pooling_block,
        upsampling_block=upsample_block,
        output_convolution_block=convolution_block,
    )

    return generator


def discriminator(depth, config):

    sub = np.concat([config["target_sub"], config["input_sub"]], axis=-1).reshape(
        (1, 1, 1, -1)
    )
    div = np.concat([config["target_div"], config["input_div"]], axis=-1).reshape(
        (1, 1, 1, -1)
    )

    def disc_norm(x):
        return

    activation = layers.LeakyReLU(0.2)

    normalization_layer = layers.Lambda(
        lambda x: K.tanh(3 * (x - sub) / (div - sub) - 1.5)
    )

    discriminator_convolution_block = dt.layers.ConvolutionalBlock(
        kernel_size=(4, 4),
        strides=1,
        activation=activation,
        instance_norm=lambda x: (
            False if x == 16 else {"axis": -1, "center": False, "scale": False},
        ),
    )

    discriminator_pooling_block = dt.layers.ConvolutionalBlock(
        kernel_size=(4, 4),
        strides=2,
        activation=activation,
        instance_norm={"axis": -1, "center": False, "scale": False},
    )

    return dt.models.convolutional(
        input_shape=[
            (None, None, 3),
            (None, None, 7),
        ],  # shape of the input
        conv_layers_dimensions=(
            16,
            32,
            64,
            128,
            256,
        ),  # number of features in each convolutional layer
        dense_layers_dimensions=(),  # number of neurons in each dense layer
        number_of_outputs=1,  # number of neurons in the final dense step (numebr of output values)
        compile=False,
        input_layer=normalization_layer,
        output_kernel_size=4,
        dense_top=False,
        convolution_block=discriminator_convolution_block,
        pooling_block=discriminator_pooling_block,
    )
