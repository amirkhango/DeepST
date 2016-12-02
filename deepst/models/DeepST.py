'''
    personal region + CPT residual modalities
'''

from __future__ import print_function
import sys
from keras.models import Sequential
from keras.layers import (
    Input,
    Activation,
    merge,
    Dense,
    Flatten,
    Reshape
)
from keras.layers.convolutional import (
    Convolution2D,
    MaxPooling2D,
    AveragePooling2D
)
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.models import Model
from keras.utils.visualize_util import plot

import numpy as np


def _shortcut(input, residual):
    return merge([input, residual], mode='sum')

# Helper to build a BN -> relu -> conv block
# This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf


def _bn_relu_conv(nb_filter, nb_row, nb_col, subsample=(1, 1), bn=False):
    def f(input):
        if bn:
            input = BatchNormalization(mode=0, axis=1)(input)
        activation = Activation('relu')(input)
        return Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample, border_mode="same")(activation)
    return f


# Builds a residual block with repeating bottleneck blocks.
def _residual_block(block_function, nb_filter, repetations=1, is_first_layer=False):
    def f(input):
        for i in xrange(repetations):
            init_subsample = (1, 1)
            # if i == 0 and not is_first_layer:
            #    init_subsample = (2, 2)
            input = block_function(nb_filter=nb_filter, init_subsample=init_subsample)(input)
        return input

    return f

# Basic 3 X 3 convolution blocks.
# Use for resnet with layers <= 34
# Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf


def _basic_block(nb_filter, init_subsample=(1, 1)):
    def f(input):
        #conv1 = _bn_relu_conv(nb_filter, 3, 3, subsample=init_subsample)(input)
        #residual = _bn_relu_conv(nb_filter, 3, 3)(conv1)
        residual = _bn_relu_conv(nb_filter, 3, 3)(input)
        # residual = _bn_relu_conv(nb_filter, 3, 3)(residual)
        return residual # _shortcut(input, residual)

    return f

def stresnet(c_conf=(2, 3, 32, 32), p_conf=(2, 3, 32, 32), t_conf=(2, 3, 32, 32), metadata_dim=8, nb_resblock=3):
    '''
    C - Temporal Closeness
    P - Period
    T - Trend
    conf = (nb_flow, seq_len, map_height, map_width)
    metadata_dim
    '''

    # main input

    inputs = []
    outputs = []
    block_fn = _basic_block
    for conf in [c_conf, p_conf, t_conf]:
        if conf is not None:
            nb_flow, seq_len, map_height, map_width = conf
            input = Input(shape=(nb_flow * seq_len, map_height, map_width))
            inputs.append(input)
            conv1 = Convolution2D(nb_filter=64, nb_row=3, nb_col=3, border_mode="same")(input)
            outputs.append(conv1)

    output = merge(outputs, mode='sum')

    block1 = _residual_block(block_fn, nb_filter=64, repetations=nb_resblock)(output)
    activation = Activation('relu')(block1)
    final_output = Convolution2D(nb_filter=nb_flow, nb_row=3, nb_col=3, border_mode="same")(activation)


    # fusion_out = Activation('relu')(fusion_out)
    if metadata_dim != None and metadata_dim > 0:
        # auxiliary input
        aux_input = Input(shape=(metadata_dim,))
        inputs.append(aux_input)

        embedding = Dense(output_dim=10)(aux_input)
        embedding = Activation('relu')(embedding)
        h1 = Dense(output_dim=nb_flow * map_height * map_width)(embedding)
        activation = Activation('relu')(h1)
        aux_out = Reshape((nb_flow, map_height, map_width))(activation)
        final_output = merge([final_output, aux_out], mode='sum')
    else:
        print('metadata_dim:', metadata_dim)

    final_output = Activation('tanh')(final_output)
    model = Model(input=inputs, output=final_output)

    return model

if __name__ == '__main__':
    model = stresnet()
    plot(model, to_file='model-resnet-v5.png', show_shapes=True)
    model.summary()
