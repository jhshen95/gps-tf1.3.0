'''This file provides VAE models'''

import tensorflow as tf
from gps.algorithm.policy_opt.tf_utils import TfMap
from gps.algorithm.policy_opt.tf_model_example import init_weights, init_bias, get_mlp_layers

def get_decoder_mlp_layers(decoder_input, policy_input, number_layers, dimension_hidden, name):
    """compute MLP with specified number of layers.
        math: sigma(Wx + b)
        for each layer, where sigma is by default relu"""
    decoder_cur_top = decoder_input
    policy_cur_top = policy_input
    weights = []
    biases = []
    for layer_step in range(0, number_layers):
        in_shape = policy_cur_top.get_shape().dims[1].value
        cur_weight = init_weights([in_shape, dimension_hidden[layer_step]], name= name + 'w_' + str(layer_step))
        cur_bias = init_bias([dimension_hidden[layer_step]], name= name + 'b_' + str(layer_step))
        weights.append(cur_weight)
        biases.append(cur_bias)
        if layer_step != number_layers-1:  # final layer has no RELU
            decoder_cur_top = tf.nn.relu(tf.matmul(decoder_cur_top, cur_weight) + cur_bias)
            policy_cur_top = tf.nn.relu(tf.matmul(policy_cur_top, cur_weight) + cur_bias)
        else:
            decoder_cur_top = tf.matmul(decoder_cur_top, cur_weight) + cur_bias
            policy_cur_top = tf.matmul(policy_cur_top, cur_weight) + cur_bias

    return decoder_cur_top, policy_cur_top, weights, biases


def tf_vae_network(dim_input=27, dim_output=7, batch_size=25, network_config=None, dim_latent=7):
    state_input = tf.placeholder('float', [None, dim_input], name='state_input')
    action_input = tf.placeholder('float', [None, dim_output], name='action_input')

    priori_mean, priori_sigma, priori_variables = \
        priori_network(state_input, dim_latent)

    latent_mean, latent_sigma, encoder_variables = \
        encoder_network(state_input, action_input, dim_latent)

    decoder_mean, decoder_sigma, policy_mean, policy_sigma, decoder_variables = \
        decoder_network(state_input, latent_mean, latent_sigma, priori_mean, priori_sigma, dim_output)

    loss_op = get_vae_loss(action_input, priori_mean, priori_sigma, latent_mean, latent_sigma, decoder_mean, decoder_sigma)

    all_variables = priori_variables + encoder_variables + decoder_variables

    return TfMap.init_from_lists([state_input, action_input, None], [policy_mean, policy_sigma], [loss_op], policy_type="vae"), all_variables, []
    
def get_kl_div(p_mean, p_sigma, q_mean, q_sigma):
    '''
    KL divergence between two diag Guassian
    '''
    log_p_sigma = tf.math.log(p_sigma)#log(|p_var|)
    log_q_sigma = tf.math.log(q_sigma)#-log(|q_var|)
    p_div_q_sigma = tf.square(tf.divide(p_sigma, q_sigma))#Tr(inv(q_sig)*p_sig)
    kl_sig_diag = log_q_sigma - log_p_sigma + p_div_q_sigma
    diff = tf.square(tf.divide(p_mean - q_mean, q_sigma))#(p_mean-q_mean)'q_var(p_mean-q_mean)
    kl_div = 0.5*tf.reduce_sum(kl_sig_diag + diff, 1)
    return kl_div

def log_like_gaussian(target_output, action_mean, action_sigma):
    '''
    log-likelyhood of diag Gaussian
    '''
    diff = tf.square(tf.divide(action_mean- target_output, action_sigma))
    diff = -0.5*tf.reduce_sum(diff, 1)
    coef = tf.reduce_sum(tf.math.log(action_sigma), 1)
    return diff - coef

def get_vae_loss(target_output, priori_mean, priori_sigma, latent_mean, latent_sigma, action_mean, action_sigma):
    #KL divergence between priori and latent variable
    kl_loss = get_kl_div(latent_mean, latent_sigma, priori_mean, priori_sigma)
    #reconstruction error: negative log-likelihood
    rec_loss = -log_like_gaussian(target_output, action_mean, action_sigma)
    loss = tf.reduce_mean(kl_loss + rec_loss)
    return loss

def priori_network(state_input, dim_output=7, network_config=None):
    '''
    given state, output the mean and (diag) sigma of priori of latent variable z
    '''
    #n_layers = 2 if 'n_layers' not in network_config else network_config['n_layers'] + 1
    #dim_hidden = (n_layers - 1) * [40] if 'dim_hidden' not in network_config else network_config['dim_hidden']
    n_layers = 3
    dim_hidden = [40, 40]
    dim_hidden.append(dim_output*2)#outputs mean and (diag) variance

    mlp_applied, weights_FC, biases_FC = get_mlp_layers(state_input, n_layers, dim_hidden, 'priori_')
    mean = mlp_applied[:, :dim_output]
    sigma = 1.001 + tf.nn.elu(mlp_applied[:, dim_output:] - 1)
    fc_vars = weights_FC + biases_FC

    return mean, sigma, fc_vars  

def encoder_network(state_input, action_input, dim_output=7, network_config=None):
    """
    given state and true action, encodes it into the approximate posterior
    """
    #n_layers = 2 if 'n_layers' not in network_config else network_config['n_layers'] + 1
    #dim_hidden = (n_layers - 1) * [40] if 'dim_hidden' not in network_config else network_config['dim_hidden']
    n_layers = 3
    dim_hidden = [40, 40]
    dim_hidden.append(dim_output*2)#outputs mean and (diag) variance

    nn_input = tf.concat([state_input, action_input], axis=1)
    mlp_applied, weights_FC, biases_FC = get_mlp_layers(nn_input, n_layers, dim_hidden, 'encoder_')
    mean = mlp_applied[:, :dim_output]
    sigma = 1.001 + tf.nn.elu(mlp_applied[:, dim_output:] - 1)
    fc_vars = weights_FC + biases_FC

    return mean, sigma, fc_vars

def decoder_network(state_input, latent_mean, latent_sigma, priori_mean, priori_sigma, \
    dim_output=7, network_config=None):
    """
    given state and latent variable, decodes action
    """   
    #n_layers = 2 if 'n_layers' not in network_config else network_config['n_layers'] + 1
    #dim_hidden = (n_layers - 1) * [40] if 'dim_hidden' not in network_config else network_config['dim_hidden']
    n_layers = 3
    dim_hidden = [40, 40]
    dim_hidden.append(dim_output*2)#outputs mean and (diag) variance

    decoder_random_input = tf.random.normal(shape=tf.shape(latent_mean))
    policy_random_input = tf.random.normal(shape=tf.shape(priori_mean))

    decoder_latent = decoder_random_input*latent_sigma + latent_mean
    policy_latent = policy_random_input*priori_sigma + priori_mean

    decoder_input = tf.concat([state_input, decoder_latent], axis=1)
    policy_input = tf.concat([state_input, policy_latent], axis=1)

    decoder_mlp_applied, policy_mlp_applied, weights_FC, biases_FC = \
        get_decoder_mlp_layers(decoder_input, policy_input, n_layers, dim_hidden, 'decoder_')

    decoder_mean = decoder_mlp_applied[:, :dim_output]
    decoder_sigma = 1.001 + tf.nn.elu(decoder_mlp_applied[:, dim_output:] - 1)

    policy_mean = policy_mlp_applied[:, :dim_output]
    policy_sigma = 1.001 + tf.nn.elu(policy_mlp_applied[:, dim_output:] - 1)

    fc_vars = weights_FC + biases_FC
    return decoder_mean, decoder_sigma, policy_mean, policy_sigma, fc_vars