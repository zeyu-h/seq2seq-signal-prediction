import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

exercise = 1  # Possible values: 1, 2, 3, or 4.

from datasets import generate_x_y_data_v1, generate_x_y_data_v2, generate_x_y_data_v3, generate_x_y_data_v4, timer

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
output_seq_length = sample_y.shape[0]
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



class Input(object):
    def __init__(self, batch_size, num_steps, data):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.input_data, self.targets = batch_producer(data, batch_size, num_steps)


# create the main model
class Model(object):
    def __init__(self, input, is_training, hidden_size, output_dim, num_layers, dropout=0.5, init_scale=0.05):
        self.is_training = is_training
        self.input_obj = input
        self.batch_size = input.batch_size
        self.num_steps = input.num_steps
        self.hidden_size = hidden_size

        # add a dropout wrapper if training
        if is_training and dropout < 1:
            inputs = tf.nn.dropout(self.input_obj.input_data, dropout)
        else:
            inputs = self.input_obj.input_data

        # set up the state storage / extraction
        # each lstm cell has two input, i.e previous output from the cell and the previous state, to load its full state
        self.init_state = tf.placeholder(tf.float32, [num_layers, 2, self.batch_size, self.hidden_size])

        # setting up this state data variable in the format required to feed it into the TensorFlow LSTM data structure
        state_per_layer_list = tf.unstack(self.init_state, axis=0)
        rnn_tuple_state = tuple(
            [tf.contrib.rnn.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
             for idx in range(num_layers)]
        )

        # create an LSTM cell to be unrolled
        cell = tf.contrib.rnn.LSTMCell(hidden_size, forget_bias=1.0)
        # add a dropout wrapper if training
        if is_training and dropout < 1:
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout)
        if num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([cell for _ in range(num_layers)], state_is_tuple=True)

        output, self.state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32, initial_state=rnn_tuple_state)
        # reshape to (batch_size * num_steps, hidden_size)
        output = tf.reshape(output, [-1, hidden_size])

        output_w = tf.Variable(tf.random_uniform([hidden_size, output_dim], -init_scale, init_scale))
        output_b = tf.Variable(tf.random_uniform([output_dim], -init_scale, init_scale))
        final_output = tf.nn.xw_plus_b(output, output_w, output_b)
        # Reshape logits to be a 3-D tensor for sequence loss
        final_output = tf.reshape(final_output, [self.batch_size, self.num_steps, output_dim])

        # Use the contrib sequence loss and average over the batches
        loss = tf.nn.l2_loss(final_output - self.input_obj.targets)
        # Update the cost
        self.cost = tf.reduce_sum(loss)

        if not is_training:
           return
        self.learning_rate = tf.Variable(0.0, trainable=False)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), 5)
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        # optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())
        # self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)

        self.new_lr = tf.placeholder(tf.float32, shape=[])
        self.lr_update = tf.assign(self.learning_rate, self.new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self.lr_update, feed_dict={self.new_lr: lr_value})