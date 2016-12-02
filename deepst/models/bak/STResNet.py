'''
    ST-ResNet
'''

from __future__ import print_function
from keras.layers import (
    Input,
    Activation,
    merge,
    Dense,
    Reshape
)
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.utils.visualize_util import plot

def _shortcut(input, residual):
    return merge([input, residual], mode='sum')

# Helper to build a BN -> relu -> conv block
# This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
def _bn_relu_conv(nb_filter, nb_row, nb_col, bn=False):
    def f(input):
        if bn:
            input = BatchNormalization(mode=0, axis=1)(input)
        activation = Activation('relu')(input)
        return Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, border_mode="same")(activation)
    return f

def _residual_unit(nb_filter, nb_row=3, nb_col=3, repetations=2):
    def f(input):
        residual = input
        for i in xrange(repetations):
            residual = _bn_relu_conv(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col)(residual)
        return _shortcut(input, residual)
    return f

def stresnet(c_conf=(2, 3, 32, 32), p_conf=(2, 3, 32, 32), t_conf=(2, 3, 32, 32), external_dim=8, nb_residual_unit=3):
    '''
    C - Temporal Closeness
    P - Period
    T - Trend
    conf = (nb_flow, seq_len, map_height, map_width)
    metadata_dim
    '''

    # main input
    main_inputs = []
    outputs = []
    for conf in [c_conf, p_conf, t_conf]:
        if conf is not None:
            nb_flow, seq_len, map_height, map_width = conf
            input = Input(shape=(nb_flow * seq_len, map_height, map_width))
            main_inputs.append(input)
            # Conv1
            output = Convolution2D(nb_filter=64, nb_row=3, nb_col=3, border_mode="same")(input)
            # [nb_residual_unit] Residual Units
            for i in range(nb_residual_unit):
                output = _residual_unit(nb_filter=64, repetations=2)(output)
            # Conv2
            activation = Activation('relu')(output)
            output = Convolution2D(nb_filter=nb_flow, nb_row=3, nb_col=3, border_mode="same")(activation)
            outputs.append(output)

    # parameter-matrix-based fusion
    if len(outputs) == 1:
        main_output = outputs[0]
    else:
        from iLayer import iLayer
        new_outputs = []
        for output in outputs:
            new_outputs.append(iLayer()(output))
        main_output = merge(new_outputs, mode='sum')

    # fusing with external component
    if external_dim != None and external_dim > 0:
        # external input
        external_input = Input(shape=(external_dim,))
        main_inputs.append(external_input)
        embedding = Dense(output_dim=10)(external_input)
        embedding = Activation('relu')(embedding)
        h1 = Dense(output_dim=nb_flow * map_height * map_width)(embedding)
        activation = Activation('relu')(h1)
        external_output = Reshape((nb_flow, map_height, map_width))(activation)
        main_output = merge([main_output, external_output], mode='sum')
    else:
        print('external_dim:', external_dim)

    main_output = Activation('tanh')(main_output)
    model = Model(input=main_inputs, output=main_output)

    return model

if __name__ == '__main__':
    model = stresnet(external_dim=28, nb_residual_unit=12)
    plot(model, to_file='ST-ResNet.png', show_shapes=True)
    model.summary()
