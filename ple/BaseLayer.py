from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#import tensorflow as tf
import tensorflow.compat.v1 as tf


class CGCLayer(tf.keras.layers.Layer):
    def __init__(self, num_experts, num_tasks, bias_init=[], l2_reg=0.001, seed=1024, **kwargs):
        """
        Model: General MMoE Layer

        Paper: Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts


        """
        super(CGCLayer, self).__init__(**kwargs)
        #self.experts_hidden_units = experts_hidden_units
        # self.num_experts = len(self.experts_hidden_units)
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.l2_reg = l2_reg
        self.seed = seed
        self.bias_init = bias_init

    def build(self, input_shape):
        """
        Args:
            input_shape: Shape tuple (tuple of integers) or list of shape tuples (one per output tensor of the layer).
        """
        #assert input_shape is not None and len(input_shape) >= 2

        shared_input_dim = int(input_shape[0][-1])
        #self.last_expert_dims = [int(dim[-1]) for dim in self.experts_hidden_units]
        self.units = int(input_shape[1][1])

        self.gate_kernels, self.gate_bias = [], []
        for i in range(self.num_tasks):
            unit = shared_input_dim
            tf.logging.info('CGCLayer: unit {}'.format(unit))
            self.gate_kernels.append(self.add_weight(name='gate_kernel_task_{}'.format(i), shape=(unit, self.num_experts),
                                                     initializer=tf.keras.initializers.he_normal(seed=self.seed),
                                                     regularizer=tf.keras.regularizers.L1L2(0, self.l2_reg), trainable=True))
            self.gate_bias.append(
                self.add_weight(name='gate_bias_task_{}'.format(i), shape=(self.num_experts, ),
                                initializer=tf.keras.initializers.Zeros() if len(self.bias_init)==0 else tf.constant_initializer(self.bias_init[i]),
                                regularizer=tf.keras.regularizers.L1L2(0, self.l2_reg), trainable=True))

        super(CGCLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """
        Args:
            inputs: list of tensors, [shared_outputs, experts_output]
                shared_outputs: (batch, dim_1)
                experts_output: (batch, dim_2, n_experts)
        returns:
            list of tensors (N, ..., dim_i)
        """
        shared_outputs, expert_outputs = inputs

        gate_outputs = []
        for i in range(self.num_tasks):
            gate_output_i = tf.keras.backend.bias_add(tf.keras.backend.dot(shared_outputs, self.gate_kernels[i]), self.gate_bias[i])
            tf.logging.info('CGCLayer: {}th gate, gate_output {}'.format(i, gate_output_i))
            gate_output_i = tf.keras.layers.Lambda(lambda x: tf.nn.softmax(x, axis=-1))(gate_output_i)
            gate_outputs.append(gate_output_i)
            for j in range(self.num_experts):
                tf.summary.scalar('{}_expert_gate_output_{}'.format(j, i), tf.reduce_mean(gate_output_i[:, j]))

        final_outputs = []
        for i, gate_output in enumerate(gate_outputs):
            expanded_gate_output = tf.keras.backend.expand_dims(gate_output, axis=1)
            tf.logging.info('CGCLayer: {}th gate_experts, expanded_gate_output {}'.format(i, expanded_gate_output))
            #weighted_expert_output = expert_outputs * tf.keras.backend.repeat_elements(expanded_gate_output, self.units, axis=1)
            if(i<self.units-1):
                expert_unit = int(expert_outputs[i].get_shape().as_list()[-1])
                print('====self.units : {}th expert_outputs, expert_outputs[i] {}'.format(self.units, expert_outputs[i].get_shape()))
                print('====expert_unit : {}th expert_outputs, expert_outputs {}'.format(expert_unit,expert_outputs.get_shape()))
                weighted_expert_output = expert_outputs[i] * tf.keras.backend.repeat_elements(expanded_gate_output, self.units, axis=1)
                weighted_expert_output_next = expert_outputs[i+1] * tf.keras.backend.repeat_elements(expanded_gate_output,self.units, axis=1)
            else:
                expert_unit = int(expert_outputs[i].get_shape().as_list()[-1])
                weighted_expert_output = expert_outputs[i] * tf.keras.backend.repeat_elements(expanded_gate_output, self.units, axis=1)
                weighted_expert_output_next = expert_outputs[i-1] * tf.keras.backend.repeat_elements(expanded_gate_output,self.units, axis=1)

            tf.logging.info('CGCLayer: {}th gate_experts, weighted_expert_output {}'.format(i, weighted_expert_output))
            #final_outputs.append(tf.keras.backend.sum(weighted_expert_output, axis=2))
            final_outputs.append(tf.keras.backend.sum(weighted_expert_output+weighted_expert_output_next, axis=2))
            tf.logging.info('CGCLayer: {}th gate_experts, final_outputs {}'.format(i, final_outputs[-1]))

        return final_outputs

    def compute_output_shape(self, input_shape):
        #assert input_shape is not None and len(input_shape) >= 2

        output_shape = list(input_shape[1])[:-1]
        output_shape = tuple(output_shape)

        return [output_shape for _ in range(self.num_tasks)]

    def get_config(self):
        config = {'num_experts': self.num_experts, 'num_tasks': self.num_tasks, 'l2_reg': self.l2_reg, 'seed': self.seed, 'bias_init':self.bias_init}
        base_config = super(CGCLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
