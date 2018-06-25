from random import randint
from numpy import array
from numpy import argmax
from numpy import zeros
import tensorflow as tf
from source.LSTM import LSTM
from source.LSTM import LSTMAttention
# from keras import Sequential
# from keras.layers import LSTM

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
def get_pair(batch_size, n_in, n_out, cardinality):
    # generate random sequence
    Xout = zeros([batch_size,n_in,cardinality])

    yout = zeros([batch_size,n_out,cardinality])
    for i in range(batch_size):
        sequence_in = generate_sequence(n_in, cardinality)
        sequence_out = sequence_in[:n_out] + [0 for _ in range(n_in - n_out)]
        # one hot encode
        X = one_hot_encode(sequence_in, cardinality)
        y = one_hot_encode(sequence_out, cardinality)
        # reshape as 3D
        Xout[i] = X.reshape((1, X.shape[0], X.shape[1]))
        yout[i] = y.reshape((1, y.shape[0], y.shape[1]))
    return Xout, yout


# configure problem
n_features = 50
n_timesteps_in = 5
n_timesteps_out = 5
n_cell = 30
batch_size = 4

x,y = get_pair(batch_size, n_timesteps_in,n_timesteps_out, n_features)

# define model
batchX_placeholder = tf.placeholder(tf.float32, [batch_size, n_timesteps_in, n_features])
# print(batchX_placeholder)
batchY_placeholder = tf.placeholder(tf.int32, [batch_size, n_timesteps_out, n_features])
Encoder = LSTM(batchX_placeholder, n_cell)
Decoder = LSTMAttention(Encoder, n_cell)
output = tf.layers.dense(inputs=Decoder, units=n_features, activation=tf.nn.softmax)




#
# train LSTM
# for epoch in range(5000):
#     # generate new random sequence
#     X, y = get_pair(n_timesteps_in, n_timesteps_out, n_features)
#     # fit model for one epoch on this sequence
#     model.fit(X, y, epochs=1, verbose=2)
# # evaluate LSTM
# total, correct = 100, 0
# for _ in range(total):
#     X, y = get_pair(n_timesteps_in, n_timesteps_out, n_features)
#     yhat = model.predict(X, verbose=0)
#     if array_equal(one_hot_decode(y[0]), one_hot_decode(yhat[0])):
#         correct += 1
# print('Accuracy: %.2f%%' % (float(correct) / float(total) * 100.0))
# # spot check some examples
# for _ in range(10):
#     X, y = get_pair(n_timesteps_in, n_timesteps_out, n_features)
#     yhat = model.predict(X, verbose=0)
#     print('Expected:', one_hot_decode(y[0]), 'Predicted', one_hot_decode(yhat[0]))