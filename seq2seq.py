import numpy as np
import random
import math

import tensorflow as tf  # Version 1.0 or 0.12
import matplotlib.pyplot as plt

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


sample_x, sample_y = generate_x_y_data(isTrain=True, batch_size=3)
print("Dimensions of the dataset for 3 X and 3 Y training examples : ")
print(sample_x.shape)
print(sample_y.shape)
print("(seq_length, batch_size, output_dim)")

# Internal neural network parameters
input_seq_length = sample_x.shape[0]  # Time series will have the same past and future (to be predicted) lenght.
output_seq_length = sample_y.shape[1]
batch_size = 50  # Low value used for live demo purposes - 100 and 1000 would be possible too, crank that up!

output_dim = input_dim = sample_x.shape[-1]  # Output dimension (e.g.: multiple signals at once, tied in time)
hidden_dim = 30 # Count of hidden neurons in the recurrent units.
layers_stacked_count = 2  # Number of stacked recurrent cells, on the neural depth axis.

# Optmizer:
learning_rate = 0.007  # Small lr helps not to diverge during training.
nb_iters = 500  # How many times we perform a training step (therefore how many times we show a batch).
lr_decay = 0.92  # default: 0.9 . Simulated annealing.
momentum = 0.5  # default: 0.0 . Momentum technique in weights update
lambda_l2_reg = 0.003  # L2 regularization of weights - avoids overfitting


# Backward compatibility for TensorFlow's version 0.12:
try:
    tf.nn.seq2seq = tf.contrib.legacy_seq2seq
    tf.nn.rnn_cell = tf.contrib.rnn
    tf.nn.rnn_cell.GRUCell = tf.contrib.rnn.GRUCell
    print("TensorFlow's version : 1.0 (or more)")
except:
    print("TensorFlow's version : 0.12")

tf.reset_default_graph()

with tf.variable_scope('Seq2seq'):
    # Encoder: inputs
    enc_inp = [
        tf.placeholder(tf.float32, shape=(None, input_dim), name="inp_{}".format(t)) for t in range(input_seq_length)
    ]

    # Decoder: expected outputs
    expected_sparse_output = [
        tf.placeholder(tf.float32, shape=(None, output_dim), name="expected_sparse_output_".format(t)) for t in range(output_seq_length)
    ]

    # Give a "GO" token to the decoder.
    # Note: we might want to fill the encoder with zeros or its own feedback rather than with "+ enc_inp[:-1]"
    dec_inp = [tf.zeros_like(enc_inp[0], dtype=np.float32, name="GO")] + enc_inp[:-1]

    # Create a `layers_stacked_count` of stacked RNNs (GRU cells here).
    cells = []
    for i in range(layers_stacked_count):
        with tf.variable_scope('RNN_{}'.format(i)):
            cells.append(tf.nn.rnn_cell.GRUCell(hidden_dim))
            # cells.append(tf.nn.rnn_cell.BasicLSTMCell(...))
    cell = tf.nn.rnn_cell.MultiRNNCell(cells)

    # Here, the encoder and the decoder uses the same cell, HOWEVER,
    # the weights aren't shared among the encoder and decoder, we have two
    # sets of weights created under the hood according to that function's def.
    dec_outputs, dec_memory = tf.contrib.legacy_seq2seq.basic_rnn_seq2seq(
        enc_inp,
        dec_inp,
        cell
    )

    # For reshaping the output dimensions of the seq2seq RNN:
    w_out = tf.Variable(tf.random_normal([hidden_dim, output_dim]))
    b_out = tf.Variable(tf.random_normal([output_dim]))

    # Final outputs: with linear rescaling for enabling possibly large and unrestricted output values.
    output_scale_factor = tf.Variable(1.0, name="Output_ScaleFactor")

    reshaped_outputs = [output_scale_factor * (tf.matmul(i, w_out) + b_out) for i in dec_outputs]


# Training loss and optimizer
with tf.variable_scope('Loss'):
    # L2 loss
    output_loss = 0
    for _y, _Y in zip(reshaped_outputs, expected_sparse_output):
        output_loss += tf.reduce_mean(tf.nn.l2_loss(_y - _Y))

    # L2 regularization (to avoid overfitting and to have a  better generalization capacity)
    reg_loss = 0
    for tf_var in tf.trainable_variables():
        if not ("Bias" in tf_var.name or "Output_" in tf_var.name):
            reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))

    loss = output_loss + lambda_l2_reg * reg_loss

with tf.variable_scope('Optimizer'):
    optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=lr_decay, momentum=momentum)
    train_op = optimizer.minimize(loss)


def train_batch(sess, batch_size):
    """
    Training step that optimizes the weights
    provided some batch_size X and Y examples from the dataset.
    """
    X, Y = generate_x_y_data(isTrain=True, batch_size=batch_size)
    feed_dict = {enc_inp[t]: X[t] for t in range(len(enc_inp))}
    feed_dict.update({expected_sparse_output[t]: Y[t] for t in range(len(expected_sparse_output))})
    _, loss_t = sess.run([train_op, loss], feed_dict)
    return loss_t

def test_batch(sess, batch_size):
    """
    Test step, does NOT optimizes. Weights are frozen by not
    doing sess.run on the train_op.
    """
    X, Y = generate_x_y_data(isTrain=False, batch_size=batch_size)
    feed_dict = {enc_inp[t]: X[t] for t in range(len(enc_inp))}
    feed_dict.update({expected_sparse_output[t]: Y[t] for t in range(len(expected_sparse_output))})
    loss_t = sess.run([loss], feed_dict)
    return loss_t[0]


def plot_loss(train_losses, test_losses):
    # Plot loss over time:
    plt.figure(figsize=(12, 6))
    plt.plot(
        np.array(range(0, len(test_losses))) / float(len(test_losses) - 1) * (len(train_losses) - 1),
        np.log(test_losses),
        label="Test loss"
    )
    plt.plot(
        np.log(train_losses),
        label="Train loss"
    )
    plt.title("Training errors over time (on a logarithmic scale)")
    plt.xlabel('Iteration')
    plt.ylabel('log(Loss)')
    plt.legend(loc='best')
    plt.show()


def training(sess):
    train_losses = []
    test_losses = []

    sess.run(tf.global_variables_initializer())
    for t in range(nb_iters + 1):
        train_loss = train_batch(sess, batch_size)
        train_losses.append(train_loss)

        if t % 10 == 0:
            # Tester
            test_loss = test_batch(sess, batch_size)
            test_losses.append(test_loss)
            print("Step {}/{}, train loss: {}, \tTEST loss: {}".format(t, nb_iters, train_loss, test_loss))

    print("Fin. train loss: {}, \tTEST loss: {}".format(train_loss, test_loss))
    return train_losses, test_losses


def predicting(sess):
    nb_predictions = 5
    print("Let's visualize {} predictions with our signals:".format(nb_predictions))

    X, Y = generate_x_y_data(isTrain=False, batch_size=nb_predictions)
    feed_dict = {enc_inp[t]: X[t] for t in range(input_seq_length)}
    outputs = np.array(sess.run([reshaped_outputs], feed_dict)[0])

    for j in range(nb_predictions):
        plt.figure(figsize=(12, 3))

        for k in range(output_dim):
            past = X[:, j, k]
            expected = Y[:, j, k]
            pred = outputs[:, j, k]

            label1 = "Seen (past) values" if k == 0 else "_nolegend_"
            label2 = "True future values" if k == 0 else "_nolegend_"
            label3 = "Predictions" if k == 0 else "_nolegend_"
            plt.plot(range(len(past)), past, "o--b", label=label1)
            plt.plot(range(len(past), len(expected) + len(past)), expected, "x--b", label=label2)
            plt.plot(range(len(past), len(pred) + len(past)), pred, "o--y", label=label3)

        plt.legend(loc='best')
        plt.title("Predictions v.s. true values")
        plt.show()



if __name__ == "__main__":
    with tf.Session() as sess:
        with timer('training'):
            train_losses, test_losses = training(sess)
        plot_loss(train_losses, test_losses)

        predicting(sess)

    print("Reminder: the signal can contain many dimensions at once.")
    print("In that case, signals have the same color.")
    print("In reality, we could imagine multiple stock market symbols evolving,")
    print("tied in time together and seen at once by the neural network.")