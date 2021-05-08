"""
Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations

Copyright (c) 2021 boostAgent, Inc
Licensed under the MIT License (see LICENSE for details)
Written by scalaboy keith
"""




from ple.BaseModel import PleOneLayerModel
#import tensorflow as tf
import tensorflow.compat.v1 as tf

params = dict(
    learning_rate=0.00005,
    l2=0.0001,
    optimizer='adam',
    hidden_units=[256, 128, 64],
    model_tag='PleOneLayer_Model',
    experts_hidden_units=[[64], [64], [64], [64], [64], [64], [64], [64]],
    sharing_hidden_units=[64, 16],
    task_hidden_units=[[16, 8], [16, 8]],
    act_fn='relu',
    l2_reg=0.001,
    dropout_rate=0.1
)

def get_mmoe_model(inputs, extra_input=None, training=None, params=dict()):
    model_tag = params['model_tag']
    tf.logging.info('======> Start to build Alps_biz model_tag {}'.format(model_tag))

    field_size = 12
    emb_dim = 8
    experts_hidden_units = params['experts_hidden_units']
    sharing_hidden_units = params['sharing_hidden_units']
    task_hidden_units = params['task_hidden_units']
    task_apply_final_act = False
    act_fn = params['act_fn']
    l2_reg = params['l2_reg']
    dropout_rate = params['dropout_rate']
    use_bn = False
    seed = 1024

    model = PleOneLayerModel(experts_hidden_units=experts_hidden_units, sharing_hidden_units=sharing_hidden_units,
                             task_hidden_units=task_hidden_units,
                             task_apply_final_act=task_apply_final_act, act_fn=act_fn, l2_reg=l2_reg,
                             dropout_rate=dropout_rate, use_bn=use_bn, seed=seed)
    output = model(inputs, training=training)
    model.summary(print_fn=tf.logging.info)
    tf.logging.info('{}, output {}'.format(model_tag, output))
    tf.logging.info('<====== Finish building Alps_biz model_tag {}'.format(model_tag))
    return output

