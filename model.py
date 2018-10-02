import sys
####################################################
# Pull pretrain model graph, if no, leave it blank #
sys.path.append("/home/seanyu/Common_tools/")
sys.path.append("/home/seanyu/Common_tools/model_nasnet/")
sys.path.append("/home/seanyu/Common_tools/model_inception_resnet")
sys.path.append("/home/seanyu/Common_tools/model_resnet/")
#####################################################

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as slimNet

from model_pnas5 import *
from resnet_inception_v2 import *
from resnet import *
#from resnet_se import *

def create_model(Hparams):
    im_w, im_h, im_c = Hparams.input_size
    
    # placeholders
    drp_holder = tf.placeholder(tf.float32)
    input1 = tf.placeholder(dtype=tf.float32, shape = (None, im_w, im_h, im_c), name = 'input1')
    y_true1 = tf.placeholder(dtype=tf.float32, shape = (None, Hparams.n_class), name='y_true1')
    
    is_training = tf.placeholder(dtype=tf.bool, shape=[])
    lr = tf.placeholder(tf.float32, shape = [])
    
    # model structs
    if Hparams.pretrain_model == 'pnas':
        with slim.arg_scope(pnasnet_large_arg_scope(batch_norm_decay=0.95)):
            _, layers_dict = build_pnasnet_large(images=input1, num_classes=Hparams.n_class)
        exclude = ['aux_7/aux_logits', 'final_layer', 'cell_stem_0/comb_iter_0/left/global_step']
        
    elif Hparams.pretrain_model == 'nas':
        with slim.arg_scope(nasnet.nasnet_large_arg_scope(batch_norm_decay=0.95)):
            _, layers_dict = nasnet.build_nasnet_large(images=input1, num_classes=Hparams.n_class)
        exclude = ['aux_11/aux_logits', 'cell_stem_0/comb_iter_0/left/global_step', 'final_layer/FC'] # nas
    
    elif Hparams.pretrain_model == 'inception_resnet':
        with slim.arg_scope(inception_resnet_v2_arg_scope(batch_norm_decay=0.95)):
            _, layers_dict = inception_resnet_v2(inputs=input1, num_classes=Hparams.n_class, is_training = is_training)
        exclude = ['InceptionResnetV2/Logits', 'InceptionResnetV2/AuxLogits'] # inception_resnet_v2
    
    elif Hparams.pretrain_model == 'resnet_50' or Hparams.pretrain_model == "resnet_50_se":
        with slim.arg_scope(resnet_utils.resnet_arg_scope(batch_norm_decay=0.95)):
            _, layers_dict = resnet_v2_50(inputs=input1, is_training=is_training)
        exclude = []
    
    elif Hparams.pretrain_model == 'resnet_101':
        with slim.arg_scope(resnet_utils.resnet_arg_scope(batch_norm_decay=0.95)):
            _, layers_dict = resnet_v2_101(inputs=input1, is_training=is_training)
        exclude = []
        
    elif Hparams.pretrain_model == 'resnet_152':
        with slim.arg_scope(resnet_utils.resnet_arg_scope(batch_norm_decay=0.95)):
            _, layers_dict = resnet_v2_152(inputs=input1, is_training=is_training)
        exclude = []
        
    else:
        print("No pretrain model")
        layers_dict = None
    
    # Exclude variables: don't restore them   
    var_list = slim.get_variables_to_restore(exclude = exclude) # get variables here, prevent take other irrelevent variables
    
    # special exclude
    var_list = [i for i in var_list if 'squeeze_and_excitation' not in i.name] # skip squeeze and exciation
    
    #global_pool
    gap = layers_dict['global_pool']
    if len(gap.shape) == 4:
        gap = tf.reduce_mean(gap, [1,2], name = "GAP_layer") # global_averaing pooling
    
    with tf.variable_scope('classifier'):        
        logits = tf.layers.dense(inputs=gap, units=Hparams.n_class, name='logits')
        prediction = tf.nn.softmax(logits)
        if Hparams.class_weight is None:
            loss1 = tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = y_true1, name = 'loss1')
        else:
            c_weights = tf.constant(Hparams.class_weight)
            c_weights = tf.gather(c_weights, tf.cast(tf.argmax(y_true1,axis= -1),tf.int32) )
            loss1 = tf.losses.sparse_softmax_cross_entropy(labels = tf.argmax(y_true1,axis = -1), 
                                                           logits = logits, 
                                                           weights = c_weights)
    
    with tf.variable_scope('ComputeLoss'):
        loss1 = tf.reduce_mean(loss1)
        global_loss = loss1
    
    ### 
    if Hparams.optimizer is 'sgd':
        optimizer = tf.train.MomentumOptimizer(lr, use_nesterov = True, momentum = 0.95)
    else:
        optimizer =  tf.train.AdamOptimizer(lr)
    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        update = optimizer.minimize(global_loss)
    
    # other
    all_vars = tf.global_variables() #tf.all_variables() # seems it will depricate after certain version of tensorflow
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    
    with tf.variable_scope('metrics'):
        correct_prediction = tf.equal(tf.argmax(prediction, axis = 1), tf.argmax(y_true1, axis = 1))
        accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # return model as a dictionary, make it easy to access when training or evaluation
    model_key = {'input': [input1],
                 'ground_truth': [y_true1],
                 'output': {'prediction1': prediction
                            },
                 'metrics': {'accuracy': accuracy_op}, # add other metrics here (for example, f1, auc)
                 'loss': [global_loss],
                 'update': update,
                 'learning_rate': lr,
                 'is_training': is_training,
                 'intializer': init,
                 'saver': saver, # keep None if no saver
                 'vars': {'partial_vars': var_list, # partial parameters for other usage (for instance, restore)
                          'all_vars': all_vars},
                 'optional': {'dropout': drp_holder, 'layers_dict':layers_dict}
                 }
    
    metric_history = {k: {'train':[], 'valid':[]} for k in list(model_key['metrics'].keys())}
    
    return model_key, metric_history