# coding: utf-8


#import tensorflow as tf
import tensorflow.compat.v1 as tf
from ple.base  import DNNLayer
from ple.BaseLayer import CGCLayer


class PleOneLayerModel(tf.keras.Model):
    """


    inputs:
        2d tensor (batch_size, dim_1)
        2d tensor (batch_size, dim_2)

    outputs:
        2d tensor (batch_size, out_dim)

    """
    def __init__(self, experts_hidden_units, sharing_hidden_units, task_hidden_units,
                 task_apply_final_act=False, act_fn='relu', l2_reg=0.001, dropout_rate=0, use_bn=False,
                 seed=1024,name='PleOneLayerModel'):
        """
        Args:
            experts_hidden_units: list of list, unit of each hidden layer in expert layers
            sharing_hidden_units: list, unit in each hidden layer
            task_hidden_units: list of list, unit of each hidden layer in task specific layers
            act_fn: string, activation function
            l2_reg: float, regularization value
            dropout_rate: float, fraction of the units to dropout.
            use_bn: boolean, if True, apply BatchNormalization in each hidden layer
            task_apply_final_act: bool
            seed: int, random value for initialization

        """
        super(PleOneLayerModel, self).__init__(name='PleOneLayerModel')
        self.num_experts = len(experts_hidden_units)
        self.num_tasks = len(task_hidden_units)

        self.gmmoe_layer = CGCLayer(num_experts=self.num_experts, num_tasks=self.num_tasks, l2_reg=l2_reg, seed=seed,name="{}_mmoe_layer".format(name))

        self.sharing_layer = DNNLayer(hidden_units=sharing_hidden_units, activation=act_fn, l2_reg=l2_reg,
                                  dropout_rate=dropout_rate, use_bn=use_bn, seed=seed,name="{}_sharing_dnn_layer".format(name))

        self.expert_layers = []
        for i, units in enumerate(experts_hidden_units):
            self.expert_layers.append(DNNLayer(hidden_units=units, activation=act_fn, l2_reg=l2_reg,
                                             dropout_rate=dropout_rate, use_bn=use_bn,
                                             apply_final_act=True, seed=seed, name="{}_expert_{}_dnn_layer".format(name,i)))

        self.task_layers = []
        for i, units in enumerate(task_hidden_units):
            self.task_layers.append(DNNLayer(hidden_units=units, activation=act_fn, l2_reg=l2_reg,
                                      dropout_rate=dropout_rate, use_bn=use_bn, apply_final_act=task_apply_final_act, seed=seed, name="{}_task_{}_dnn_layer".format(name,i)))

    def call(self, inputs, training=None):
        """
        Args:
            inputs: 2d tensor (batch_size, dim_1), deep features

        Returns:
            list of 2d tensor (batch_size, out_dim)

        """
        expert_inputs = self.sharing_layer(inputs, training=training)
        tf.logging.info('CGCLayer: expert_inputs {}'.format(expert_inputs))

        expert_outputs_list = []
        for i in range(self.num_experts):
            expert_outputs_i = self.expert_layers[i](expert_inputs, training=training)
            expert_outputs_i = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(expert_outputs_i)
            tf.logging.info('CGCLayer: {}th exprt, expert_output {}'.format(i, expert_outputs_i))
            expert_outputs_list.append(expert_outputs_i)

        expert_outputs = tf.keras.layers.Concatenate(axis=-1)(expert_outputs_list)
        tf.logging.info('CGCLayer: expert_outputs {}'.format(expert_outputs))

        task_inputs = self.gmmoe_layer([expert_inputs, expert_outputs])

        task_outputs = []
        for i in range(self.num_tasks):
            task_outputs.append(self.task_layers[i](task_inputs[i], training=training))
            tf.logging.info('CGCLayer: {}th task, task_output {}'.format(i, task_outputs[-1]))

        return task_outputs

