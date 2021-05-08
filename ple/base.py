# coding: utf-8
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
#import tensorflow as tf

class DNNLayer(tf.keras.layers.Layer):


    def __init__(self, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False, apply_final_act=True, seed=1024, **kwargs):

        self.hidden_units = hidden_units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.seed = seed
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        self.apply_final_act = apply_final_act
        super(DNNLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        input_size = input_shape[-1]
        hidden_units = [int(input_size)] + list(self.hidden_units)

        self.kernels = [self.add_weight(name='kernel' + str(i),
                                        shape=[hidden_units[i], hidden_units[i + 1]],
                                        initializer=tf.keras.initializers.he_normal(seed=self.seed),
                                        regularizer=tf.keras.regularizers.l2(self.l2_reg),
                                        trainable=True)
                        for i in range(len(self.hidden_units))]
        self.bias = [self.add_weight(name='bias' + str(i),
                                     shape=[self.hidden_units[i], ],
                                     initializer=tf.keras.initializers.Zeros(),
                                     trainable=True)
                     for i in range(len(self.hidden_units))]

        if self.use_bn:
            self.bn_layers = [tf.keras.layers.BatchNormalization(name='bn_layer_{}'.format(i)) for i in range(len(self.hidden_units))]

        if self.dropout_rate is not None and self.dropout_rate > 0:
            self.dropout_layers = [tf.keras.layers.Dropout(self.dropout_rate,
                                                           seed=self.seed + i, name='dropout_layer_{}'.format(i))
                                   for i in range(len(self.hidden_units))]

        self.activation_layers = [tf.keras.layers.Activation(self.activation, name='act_layer_{}'.format(i)) for i in range(len(self.hidden_units))]

        super(DNNLayer, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        deep_input = inputs

        for i in range(len(self.hidden_units)):
            fc = tf.nn.bias_add(tf.tensordot(deep_input, self.kernels[i], axes=(-1, 0)), self.bias[i])
            if self.use_bn:
                fc = self.bn_layers[i](fc, training=training)

            if i < len(self.hidden_units)-1 or self.apply_final_act:
                fc = self.activation_layers[i](fc)

            if self.dropout_rate is not None and self.dropout_rate > 0:
                fc = self.dropout_layers[i](fc, training=training)
            deep_input = fc

        return deep_input

    def compute_output_shape(self, input_shape):
        if len(self.hidden_units) > 0:
            shape = input_shape[:-1] + (self.hidden_units[-1],)
        else:
            shape = input_shape

        return tuple(shape)

    def get_config(self, ):
        config = {
            'activation': self.activation,
            'hidden_units': self.hidden_units,
            'l2_reg': self.l2_reg,
            'use_bn': self.use_bn,
            'dropout_rate': self.dropout_rate,
            'apply_final_act': self.apply_final_act,
            'seed': self.seed
        }
        base_config = super(DNNLayer, self).get_config()
        config.update(base_config)
        return config
