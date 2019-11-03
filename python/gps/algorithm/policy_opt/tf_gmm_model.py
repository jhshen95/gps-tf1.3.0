""" This file provides an example tensorflow network used to define a Gaussian Mixture Model policy. """

import tensorflow as tf
from gps.algorithm.policy_opt.tf_utils import TfMap
from gps.algorithm.policy_opt.tf_model_example import init_weights, init_bias, get_input_layer, get_mlp_layers
import numpy as np


def get_normal_prob(mean, pre, action, n_comp):
    action = tf.tile(tf.expand_dims(action, 1), [1, n_comp, 1])
    result = tf.multiply(action- mean, pre)
    result = tf.square(result)
    result = -0.5*tf.reduce_sum(result, axis=2)
    log_coef = tf.reduce_sum(tf.log(pre), axis=2)
    log_result = log_coef + result
    return log_result

def logsumexp(comp):
    '''
    calculate log(sum(exp(x_i))):
    first extract the maximum of x_i(denoted as max_comp), then calculate logsumexp(x-max_comp) + max_comp
    '''
    max_comp = tf.reduce_max(comp, 1, keep_dims=True)
    comp = comp - max_comp
    result = tf.log(tf.reduce_sum(tf.exp(comp), 1)) + tf.squeeze(max_comp, 1)
    return result

def get_gmm_loss_layer(weight, mean, pre, action, n_comp, dim_output):
    #TODO: consider the precision of ground truth, use samples instead of mean
    log_result = get_normal_prob(mean, pre, action, n_comp)
    log_weight = tf.log(weight)
    log_comp = log_weight+log_result
    log_like = -logsumexp(log_result)

    loss = tf.reduce_mean(log_like) + tf.log(2*3.14)*dim_output/2
    return loss


def get_gmm_coef(output, dim_output, n_comp):
    n_dims = dim_output*n_comp
    weight, mean, pre = \
        output[:, :n_comp], output[:, n_comp:n_comp+n_dims], output[:, n_comp+n_dims:]
    #reshape [batch, n_comp*dim_out] to [batch, n_comp, dim_out]
    mean = tf.reshape(mean, [-1, n_comp, dim_output])
    pre = tf.reshape(pre, [-1, n_comp, dim_output])
    #normalize weight
    weight = tf.nn.softmax(weight)
    #pre should be positive
    pre = tf.exp(pre)
    return weight, mean, pre

def tf_gmm_network(dim_input=27, dim_output=7, batch_size=25, n_comp=3, network_config=None):
    n_layers = 2 if 'n_layers' not in network_config else network_config['n_layers'] + 1
    dim_hidden = (n_layers - 1) * [40] if 'dim_hidden' not in network_config else network_config['dim_hidden']
    #for each component, outputs weight(dim 1), mean(dim_output) and diag_covar (dim_output)
    dim_hidden.append(n_comp*(1+dim_output*2))
    
    nn_input, action, precision = get_input_layer(dim_input, dim_output)
    mlp_applied, weights_FC, biases_FC = get_mlp_layers(nn_input, n_layers, dim_hidden)
    fc_vars = weights_FC + biases_FC
    weight, mean, pre = get_gmm_coef(mlp_applied, dim_output, n_comp)

    loss_out = get_gmm_loss_layer(weight=weight, mean=mean, pre=pre, \
        action=action, n_comp=n_comp, dim_output=dim_output)

    return TfMap.init_from_lists([nn_input, action, precision], [weight, mean, pre], [loss_out], policy_type="gmm"), fc_vars, []
