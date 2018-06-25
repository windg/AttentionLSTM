# http://colah.github.io/posts/2015-08-Understanding-LSTMs/
from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from random import randint
from numpy import array
from numpy import argmax



# generate a sequence of random integers
def generate_sequence(length, n_unique):
    return [randint(0, n_unique - 1) for _ in range(length)]


# one hot encode sequence
def one_hot_encode(sequence, n_unique):
    encoding = list()
    for value in sequence:
        vector = [0 for _ in range(n_unique)]
        vector[value] = 1
        encoding.append(vector)
    return array(encoding)


# decode a one hot encoded string
def one_hot_decode(encoded_seq):
    return [argmax(vector) for vector in encoded_seq]


# prepare data for the LSTM
def get_pair(n_in, n_out, cardinality):
    # generate random sequence
    sequence_in = generate_sequence(n_in, cardinality)
    sequence_out = sequence_in[:n_out] + [0 for _ in range(n_in - n_out)]
    # one hot encode
    X = one_hot_encode(sequence_in, cardinality)
    y = one_hot_encode(sequence_out, cardinality)
    # reshape as 3D
    X = X.reshape((1, X.shape[0], X.shape[1]))
    y = y.reshape((1, y.shape[0], y.shape[1]))
    return X, y

def weight_variable(shape):
    # Random initial values
    initial = np.random.rand(shape[0], shape[1])
    return tf.Variable(initial,dtype=tf.float32)
def bias_variable(shape):
    initial = np.zeros((1, shape))
    return tf.Variable(initial,dtype=tf.float32)
def LSTMAttention(input, cells):
    batchsize, timesteps, input_dim = input.get_shape().as_list()

    inputs_series = tf.unstack(input, axis=1)
    # Variables for LSTM
    Wf = weight_variable(shape=[input_dim + cells, cells])
    bf = bias_variable(shape=cells)
    Wi = weight_variable(shape=[input_dim + cells, cells])
    bi = bias_variable(shape=cells)
    Wc = weight_variable(shape=[input_dim + cells, cells])
    bc = bias_variable(shape=cells)
    Wo = weight_variable(shape=[input_dim + cells, cells])
    bo = bias_variable(shape=cells)
    # Variable for attention model
    Va = weight_variable(shape=[input_dim,1])
    Wa = weight_variable(shape=[cells, input_dim])
    ba = bias_variable(shape=input_dim)
    Ua = weight_variable(shape=[input_dim, input_dim])
    # print(input_dim)
    embed = tf.reshape(input,[-1, input_dim])
    # print(embed.shape)
    embed = tf.matmul(embed, Ua)
    # print(embed.shape)
    # embed = tf.reshape(embed,[batch_size,timesteps,input_dim])
    # print(embed.shape)


    init_output = tf.placeholder(tf.float32, [batchsize, cells])
    init_state = tf.placeholder(tf.float32, [batchsize, cells])
    current_state = init_state
    h = init_output
    output = tf.expand_dims(init_output, axis=1)
    for current_input in inputs_series:
        # print(type(current_input))
        # current_input = tf.reshape(current_input, [batch_size, n_features])
        expanded_state = tf.tile(current_state, [timesteps,1])

        e = tf.tanh(tf.matmul(expanded_state, Wa) + embed)

        e = tf.matmul(e,tf.tile(Va, multiples=[1, input_dim]))
        e = tf.reshape(e,[batchsize,timesteps,-1])

        a = tf.nn.softmax(e,dim=1)
        # print(a.shape)
        c = tf.reduce_sum(tf.multiply(a,input),axis=1)
        # print(c.shape)
        stacked_input_h = tf.concat([c, h], axis=1)

        f = tf.sigmoid(tf.matmul(stacked_input_h, Wf) + bf)
        i = tf.sigmoid(tf.matmul(stacked_input_h, Wi) + bi)
        C_tilda = tf.tanh(tf.matmul(stacked_input_h, Wc) + bc)
        current_state = tf.multiply(f, current_state) + tf.multiply(i, C_tilda)
        o = tf.sigmoid(tf.matmul(stacked_input_h, Wo) + bo)
        h = tf.multiply(o, tf.tanh(current_state))
        output = tf.concat([output, tf.expand_dims(h, axis=1)], axis=1)
    return output[:,1:,:]


def LSTM(input, cells, return_sequences=False):
    batchsize, timesteps, input_dim = input.shape
    # print(input)
    inputs_series = tf.unstack(input, axis=1)
    # Variables for LSTM

    Wf = weight_variable(shape=[input_dim+cells , cells])
    bf = bias_variable(shape=cells)
    Wi = weight_variable(shape=[input_dim+cells, cells])
    bi = bias_variable(shape=cells)
    Wc = weight_variable(shape=[input_dim+cells, cells])
    bc = bias_variable(shape=cells)
    Wo = weight_variable(shape=[input_dim+cells, cells])
    bo = bias_variable(shape=cells)

    init_output = tf.placeholder(tf.float32,[batchsize, cells])
    init_state = tf.placeholder(tf.float32,[batchsize, cells])
    current_state = init_state
    h = init_output
    output = tf.expand_dims(init_output,axis=1)
    for current_input in inputs_series:
        # print(current_input.shape)
        # current_input = tf.reshape(current_input, [batch_size, n_features])
        stacked_input_h = tf.concat([current_input, h],axis=1)

        f = tf.sigmoid(tf.matmul(stacked_input_h, Wf) + bf)
        i = tf.sigmoid(tf.matmul(stacked_input_h, Wi) + bi)
        C_tilda  = tf.tanh(tf.matmul(stacked_input_h, Wc) + bc)
        current_state = tf.multiply(f, current_state) + tf.multiply(i, C_tilda)
        o = tf.sigmoid(tf.matmul(stacked_input_h, Wo) + bo)
        h = tf.multiply(o, tf.tanh(current_state))
        output = tf.concat([output, tf.expand_dims(h, axis=1)],axis = 1)
    return output[:,1:,:]

n_features = 50
n_timesteps_in = 5
n_timesteps_out = 5
n_cell = 30
batch_size = 4
#
# # X, y = get_pair(n_timesteps_in, n_timesteps_out, n_features)
# # print(X.shape)
batchX_placeholder = tf.placeholder(tf.float32, [batch_size, n_timesteps_in, n_features])
# print(batchX_placeholder)
batchY_placeholder = tf.placeholder(tf.int32, [batch_size, n_timesteps_out, n_features])

mdl = LSTM(batchX_placeholder, n_cell)
print(mdl)