from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from random import randint
from numpy import array
from numpy import argmax

n_features = 50
n_timesteps_in = 5
n_timesteps_out = 5

num_epochs = 100
total_series_length = 50000
truncated_backprop_length = 15
state_size = 4
num_class = 2
echo_step = 3
batch_size = 4
num_batches = total_series_length//batch_size//truncated_backprop_length

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
    initial = np.zeros((1,shape))
    return tf.Variable(initial,dtype=tf.float32)

def LSTM(input, cells):
    timesteps, input_dim = input.shape
    # print(input)
    inputs_series = tf.unstack(input, axis=0)
    Wf = weight_variable(shape=[cells,input_dim])

    bf = bias_variable(shape=cells)
    for current_input in inputs_series:

        current_input = tf.reshape(current_input, [n_features, 1])

        mdl = tf.matmul(Wf,current_input)
    return mdl


# X, y = get_pair(n_timesteps_in, n_timesteps_out, n_features)
# print(X.shape)
batchX_placeholder = tf.placeholder(tf.float32, [n_timesteps_in, n_features])

batchY_placeholder = tf.placeholder(tf.int32, [n_timesteps_out, n_features])

mdl = LSTM(batchX_placeholder, 3)
print(mdl)