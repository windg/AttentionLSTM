# http://colah.github.io/posts/2015-08-Understanding-LSTMs/
from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from random import randint
from numpy import array
from numpy import argmax

class EDA:
    def __init__(self, input, encoder_ncell, decoder_ncell, plot=False):
        input
        self.batchsize, self.timesteps, self.input_dim = input.get_shape().as_list()
        Encoder = self._Encoder(input, encoder_ncell)
        Decoder = self._Decoder(Encoder, decoder_ncell)
        self.output = tf.layers.dense(inputs=Decoder, units=self.input_dim, activation=tf.nn.softmax)

    def plot(loss_list):

        plt.cla()
        plt.plot(loss_list)

        plt.draw()
        plt.pause(0.0001)

    def _Encoder(self, input, cells):
        batchsize, timesteps, input_dim = input.shape

        # Variables for Encoder
        self.En_Wf = tf.get_variable(name='En_Wf', shape=[self.input_dim + cells, cells])
        self.En_bf = tf.get_variable(name='En_bf', shape=cells)
        self.En_Wi = tf.get_variable(name='En_Wi', shape=[self.input_dim + cells, cells])
        self.En_bi = tf.get_variable(name='En_bi', shape=cells)
        self.En_Wc = tf.get_variable(name='En_Wc', shape=[self.input_dim + cells, cells])
        self.En_bc = tf.get_variable(name='En_bc', shape=cells)
        self.En_Wo = tf.get_variable(name='En_Wo', shape=[self.input_dim + cells, cells])
        self.En_bo = tf.get_variable(name='En_bo', shape=cells)
        # print(input)
        inputs_series = tf.unstack(input, axis=1)
        # Variables for LSTM
        init_output = tf.zeros([batchsize, cells])
        init_state = tf.zeros([batchsize, cells])
        current_state = init_state
        h = init_output
        output = tf.expand_dims(init_output, axis=1)
        for current_input in inputs_series:
            stacked_input_h = tf.concat([current_input, h], axis=1)

            f = tf.sigmoid(tf.matmul(stacked_input_h, self.En_Wf) + self.En_bf)
            i = tf.sigmoid(tf.matmul(stacked_input_h, self.En_Wi) + self.En_bi)
            C_tilda = tf.tanh(tf.matmul(stacked_input_h, self.En_Wc) + self.En_bc)
            current_state = tf.multiply(f, current_state) + tf.multiply(i, C_tilda)
            o = tf.sigmoid(tf.matmul(stacked_input_h, self.En_Wo) + self.En_bo)
            h = tf.multiply(o, tf.tanh(current_state))
            output = tf.concat([output, tf.expand_dims(h, axis=1)], axis=1)
        return output[:, 1:, :]


    def _Decoder(self, input, cells):
        batchsize, timesteps, input_dim = input.get_shape().as_list()

        inputs_series = tf.unstack(input, axis=1)
        #
        # Variables for Decoder
        self.De_Wf = tf.get_variable(name='De_Wf', shape=[input_dim + cells, cells])
        self.De_bf = tf.get_variable(name='De_bf', shape=cells)
        self.De_Wi = tf.get_variable(name='De_Wi', shape=[input_dim + cells, cells])
        self.De_bi = tf.get_variable(name='De_bi', shape=cells)
        self.De_Wc = tf.get_variable(name='De_Wc', shape=[input_dim + cells, cells])
        self.De_bc = tf.get_variable(name='De_bc', shape=cells)
        self.De_Wo = tf.get_variable(name='De_Wo', shape=[input_dim + cells, cells])
        self.De_bo = tf.get_variable(name='De_bo', shape=cells)
        # Variable for attention model
        self.Va = tf.get_variable(name='Va', shape=[input_dim, 1])
        self.Wa = tf.get_variable(name='Wa', shape=[cells, input_dim])
        self.ba = tf.get_variable(name='ba', shape=input_dim)
        self.Ua = tf.get_variable(name='Ua', shape=[input_dim, input_dim])
        embed = tf.reshape(input,[-1, input_dim])
        embed = tf.matmul(embed, self.Ua)

        init_output = tf.zeros([batchsize, cells])
        init_state = tf.zeros([batchsize, cells])
        current_state = init_state
        h = init_output
        output = tf.expand_dims(init_output, axis=1)
        for current_input in inputs_series:
            expanded_state = tf.tile(current_state, [timesteps,1])

            e = tf.tanh(tf.matmul(expanded_state, self.Wa) + embed)

            e = tf.matmul(e, tf.tile(self.Va, multiples=[1, input_dim]))
            e = tf.reshape(e, [batchsize,timesteps,-1])

            a = tf.nn.softmax(e,dim=1)
            c = tf.reduce_sum(tf.multiply(a,input),axis=1)
            stacked_input_h = tf.concat([c, h], axis=1)

            f = tf.sigmoid(tf.matmul(stacked_input_h, self.De_Wf) + self.De_bf)
            i = tf.sigmoid(tf.matmul(stacked_input_h, self.De_Wi) + self.De_bi)
            C_tilda = tf.tanh(tf.matmul(stacked_input_h, self.De_Wc) + self.De_bc)
            current_state = tf.multiply(f, current_state) + tf.multiply(i, C_tilda)
            o = tf.sigmoid(tf.matmul(stacked_input_h, self.De_Wo) + self.De_bo)
            h = tf.multiply(o, tf.tanh(current_state))
            output = tf.concat([output, tf.expand_dims(h, axis=1)], axis=1)
        return output[:,1:,:]


def LSTM(input, cells, return_sequences=False):
    batchsize, timesteps, input_dim = input.shape
    # print(input)
    inputs_series = tf.unstack(input, axis=1)
    # Variables for LSTM

    Wf = tf.get_variable(name='Wf', shape=[input_dim+cells , cells])
    bf = tf.get_variable(name='bf', shape=cells)
    Wi = tf.get_variable(name='Wi', shape=[input_dim+cells, cells])
    bi = tf.get_variable(name='bi', shape=cells)
    Wc = tf.get_variable(name='Wc', shape=[input_dim+cells, cells])
    bc = tf.get_variable(name='bc', shape=cells)
    Wo = tf.get_variable(name='Wo', shape=[input_dim+cells, cells])
    bo = tf.get_variable(name='bo', shape=cells)
    init_output = tf.zeros([batchsize, cells])
    init_state = tf.zeros([batchsize, cells])
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
if __name__ == '__main__':
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