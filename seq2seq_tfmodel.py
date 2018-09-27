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
lambda_l2_reg = 0.0003

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
    global_step = tf.Variable(
        initial_value=0,
        name="global_step",
        trainable=False,
        collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

    with tf.variable_scope('Seq2seq'):
        encoder_inputs = tf.placeholder(shape=(None, None, input_dim), dtype=tf.float32, name='encoder_inputs')
        decoder_targets = tf.placeholder(shape=(None, None, output_dim), dtype=tf.float32, name='decoder_targets')

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
                                                #sequence_length=input_seq_len,     #may not need to set it in this example
                                                dtype=tf.float32,
                                                time_major=True)

            encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

            encoder_final_state_c = tf.concat(
                (encoder_fw_final_state[0], encoder_bw_final_state[0]), 1)

            encoder_final_state_h = tf.concat(
                (encoder_fw_final_state[1], encoder_bw_final_state[1]), 1)

            encoder_final_state = tf.contrib.rnn.LSTMStateTuple(
                c=encoder_final_state_c,
                h=encoder_final_state_h
            )

        EOS = 0
        eos_time_slice = tf.ones([batch_size, output_dim], dtype=tf.int32, name='EOS')*EOS

        #fixed output lenght, therefore is not actually used
        PAD = 1
        pad_time_slice = tf.zeros([batch_size], dtype=tf.int32, name='PAD')*PAD

        # output seq plus one EOS at the beginning
        decoder_lengths = output_seq_len + 1

        weights = {
            'out': tf.get_variable('Weights_out',
                                   shape=[hidden_dim, output_dim],
                                   dtype=tf.float32,
                                   initializer=tf.truncated_normal_initializer()),
        }
        biases = {
            'out': tf.get_variable('Biases_out',
                                   shape=[output_dim],
                                   dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.)),
        }

        def loop_fn_initial():
            initial_elements_finished = (0 >= decoder_lengths)  # all False at the initial step
            initial_input = eos_time_slice
            initial_cell_state = encoder_final_state
            initial_cell_output = None
            initial_loop_state = None  # we don't need to pass any additional information
            return (initial_elements_finished,
                    initial_input,
                    initial_cell_state,
                    initial_cell_output,
                    initial_loop_state)

        def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):

            def get_next_input():
                next_input = tf.matmul(previous_output, weights['out']) + biases['out']
                return next_input

            elements_finished = (time >= decoder_lengths)  # this operation produces boolean tensor of [batch_size]
            # defining if corresponding sequence has ended

            finished = tf.reduce_all(elements_finished)  # -> boolean scalar
            input = tf.cond(finished, lambda: pad_time_slice, get_next_input)
            state = previous_state
            output = previous_output
            loop_state = None

            return (elements_finished,
                    input,
                    state,
                    output,
                    loop_state)


        def loop_fn(time, previous_output, previous_state, previous_loop_state):
            if previous_state is None:  # time == 0
                assert previous_output is None and previous_state is None
                return loop_fn_initial()
            else:
                return loop_fn_transition(time, previous_output, previous_state, previous_loop_state)

        decoder_cell = get_rnn_cell()
        decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(decoder_cell, loop_fn)
        decoder_outputs = decoder_outputs_ta.stack()


        # Training loss and optimizer
        with tf.variable_scope('Loss'):
            # L2 loss
            output_loss = 0
            for _y, _Y in zip(decoder_outputs, decoder_targets):
                output_loss += tf.reduce_mean(tf.pow(_y - _Y, 2))

            # L2 regularization for weights and biases
            reg_loss = 0
            for tf_var in tf.trainable_variables():
                if 'Biases_' in tf_var.name or 'Weights_' in tf_var.name:
                    reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))
            loss = output_loss + lambda_l2_reg * reg_loss


        with tf.variable_scope('Optimizer'):
            optimizer = tf.contrib.layers.optimize_loss(
                loss=loss,
                learning_rate=learning_rate,
                global_step=global_step,
                optimizer='Adam',
                clip_gradients=GRADIENT_CLIPPING)

        saver = tf.train.Saver
        return dict(
            enc_inp=encoder_inputs,
            target_seq=decoder_targets,
            train_op=optimizer,
            loss=loss,
            saver=saver,
            reshaped_outputs=decoder_outputs
        )


def train(save_path):
    train_losses = []
    val_losses = []

    batch_input, batch_output = generate_x_y_data(isTrain=True,
                                                  batch_size=1,
                                                  input_seq_len=input_seq_len,
                                                  output_seq_len=output_seq_len)
    input_dim = batch_input.shape[-1]
    output_dim = batch_output.shape[-1]

    rnn_model = build_graph(feed_previous=False, input_dim=input_dim, output_dim=output_dim)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        for i in range(nb_iters):
            batch_input, batch_output = generate_x_y_data(isTrain=True,
                                                          batch_size=batch_size,
                                                          input_seq_len=input_seq_len,
                                                          output_seq_len=output_seq_len)
            input_dim = batch_input.shape[-1]
            output_dim = batch_output.shape[-1]

            feed_dict = {rnn_model['enc_inp'][t]: batch_input[t, :].reshape(-1, input_dim) for t in range(input_seq_len)}
            feed_dict.update(
                {rnn_model['target_seq'][t]: batch_output[t, :].reshape(-1, output_dim) for t in range(output_seq_len)})

            _, loss_t = sess.run([rnn_model['train_op'], rnn_model['loss']], feed_dict)
            train_losses.append(loss_t)

            if i % 10 == 0:
                print("Step {}/{}, train loss: {}".format(i, nb_iters, loss_t))

        temp_saver = rnn_model['saver']()
        save_path = temp_saver.save(sess, save_path)
    print("Checkpoint saved at: ", save_path)
    return


if __name__ == "__main__":
    save_path = os.path.join('.\seq2seq_cp', 'univariate_ts_model0')
    with timer('training'):
        train(save_path)