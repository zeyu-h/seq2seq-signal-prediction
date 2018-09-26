#based on https://github.com/ematvey/tensorflow-seq2seq-tutorials

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.contrib import rnn
import copy

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

exercise = 2  # Possible values: 1, 2, 3, or 4.

from data_util import generate_x_y_data_v1, generate_x_y_data_v2, generate_x_y_data_v3, generate_x_y_data_v4, timer

# We choose which data function to use below, in function of the exericse.
if exercise == 1:
    generate_x_y_data = generate_x_y_data_v1
if exercise == 2:
    generate_x_y_data = generate_x_y_data_v2
if exercise == 3:
    generate_x_y_data = generate_x_y_data_v3
if exercise == 4:
    generate_x_y_data = generate_x_y_data_v4

## Parameters
learning_rate = 0.01
lambda_l2_reg = 0.003

## Network Parameters
# length of input signals
input_seq_len = 60
# length of output signals
output_seq_len = 20
# size of LSTM Cell
hidden_dim = 32
# num of stacked lstm layers
num_stacked_layers = 2
#number of iteration in training
nb_iters = 200
#batch size
batch_size = 32
# gradient clipping - to avoid gradient exploding
GRADIENT_CLIPPING = 2.5


def build_graph(feed_previous=False, input_dim=1, output_dim=1):
    with tf.variable_scope('Seq2seq'):
        encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
        decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')

        def get_rnn_cell():
            with tf.variable_scope('LSTMCell'):
                cells = []
                for i in range(num_stacked_layers):
                    with tf.variable_scope('RNN_{}'.format(i)):
                        cells.append(tf.contrib.rnn.LSTMCell(hidden_dim))
                cell = tf.contrib.rnn.MultiRNNCell(cells)
            return cell

        encoder_cell = get_rnn_cell()
        with tf.variable_scope('Encoder'):
            ((encoder_fw_outputs,
                encoder_bw_outputs),
               (encoder_fw_final_state,
                encoder_bw_final_state)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,
                                                cell_bw=encoder_cell,
                                                inputs=encoder_inputs,
                                                sequence_length=input_seq_len,
                                                dtype=tf.float32, time_major=True)

            encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

            encoder_final_state_c = tf.concat(
                (encoder_fw_final_state.c, encoder_bw_final_state.c), 1)

            encoder_final_state_h = tf.concat(
                (encoder_fw_final_state.h, encoder_bw_final_state.h), 1)

            encoder_final_state = tf.contrib.rnn.LSTMStateTuple(
                c=encoder_final_state_c,
                h=encoder_final_state_h
            )

        decoder_cell = get_rnn_cell()
        with tf.variable_scope('Decoder'):
