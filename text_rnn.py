from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np

from tfdl.core import Model


class TextRNN(Model):
    def _build_graph(self):
        """
        Construct dynamic multi-layer rnn for text classification.
        DO NOT use 0 as label due to loss masking.
        
        Args
        --------
        layer_dims: a tuple that defines the dimension of the hidden layer for each layer
        cell: {'lstm', 'gru'}

        Inputs
        --------
        input_x, y: batch of indice sentence,
        input_lengths: list of lengths of each sentence in the input batch, e.g. [4,2,23,12,2]
        """
        self.input_x = self.create_input(tf.int32, [None, self.config.seq_len], name="input_x")
        self.input_y = self.create_input(tf.int32, [None, self.config.seq_len], name="input_y")
        self.input_lengths = self.create_input(tf.int32, [None], name="input_lengths")
        self.dropout_keep_prob = self.create_input(tf.float32, name="dropout_keep_prob")
        self.split_id = self.config.split_id

        # Embedding layer
        with tf.device('/cpu:0'), tf.variable_scope("embedding"):
            W = tf.get_variable("W", shape=[self.config.vocab_size, self.config.embedding_dim],
                                initializer=tf.random_uniform_initializer(-1, 1))
            # shape [None, sequence_length, embedding_dim]
            embedded = tf.nn.embedding_lookup(W, self.input_x)

        with tf.name_scope('rnn_cells'):
            if self.config.cell_type == 'lstm':
                cells = [tf.contrib.rnn.LSTMCell(num_units=layer_dim, state_is_tuple=True)
                         for layer_dim in self.config.layer_dims]
            elif self.config.cell_type == 'gru':
                cells = [tf.contrib.rnn.GRUCell(num_units=layer_dim, state_is_tuple=True)
                         for layer_dim in self.config.layer_dims]
                cells = [tf.contrib.rnn.DropoutWrapper(cell=cell,
                                                       output_keep_prob=self.dropout_keep_prob)
                         for cell in cells]
            cell = tf.contrib.rnn.MultiRNNCell(cells=cells, state_is_tuple=True)

            self.outputs, self.last_states = tf.nn.dynamic_rnn(
                cell=cell,
                dtype=tf.float32,
                sequence_length=self.input_lengths,  # accept a list of lengths
                inputs=embedded  # zero padded
            )  # outputs shape [batch, max_time, last_hidden_dim]
            self.last_c_h = tf.reshape(self.last_states,
                                       [-1, 2 * self.config.layer_dims[-1]])
            # self.config.output_ops.append('last_c_h')

        with tf.name_scope('output'):
            W = tf.get_variable("W", shape=[self.config.layer_dims[-1], self.config.vocab_size],
                                initializer=tf.random_normal_initializer())
            b = tf.get_variable("b", shape=[self.config.vocab_size],
                                initializer=tf.constant_initializer(0.0))

            # More efficient for calculating loss
            # flattened [batch x max_time, last layer hidden size]
            outputs_flatten = tf.reshape(self.outputs, [-1, self.config.layer_dims[-1]])
            self.scores_flatten = tf.nn.xw_plus_b(outputs_flatten, W, b)
            self.scores = tf.reshape(self.scores_flatten,
                                     [-1, self.config.seq_len, self.config.vocab_size])


        self.probs = tf.nn.softmax(self.scores, name="probs")
        self.preds = tf.argmax(self.probs, axis=2, name="preds")
        self.config.output_ops.append('probs')
        self.config.output_ops.append('preds')

        self.split_probs = self.probs[:, :, self.split_id]
        self.predicted_split = tf.argmax(self.split_probs, axis=1, name="predicted_split")  # [batch]
        self.config.output_ops.append('predicted_split')
        self.split_probs_max = tf.reduce_max(self.split_probs, reduction_indices=1, name="split_probs_max")
        self.config.output_ops.append('split_probs_max')

        # TODO: Try targeted trainig - train only the SPLIT position

        y_flat = tf.reshape(self.input_y, [-1])
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.scores_flatten,
                                                                    labels=y_flat)
        # Mask the loss (tf.sign(0) == 0)
        mask = tf.sign(tf.to_float(y_flat))
        # Bring back [B, T] shape
        masked_losses = tf.reshape(mask * losses, tf.shape(self.input_y))
        # Loss has to be calculated in two steps as inputs are padded
        mean_loss_by_example = tf.div(tf.reduce_sum(masked_losses, reduction_indices=1),
                                        tf.cast(self.input_lengths, tf.float32))


        # loss
        self.loss = tf.reduce_mean(mean_loss_by_example, name="loss")
        # Summary
        self.loss_summary = tf.summary.scalar("loss", self.loss)
        # self.metric_summaries = tf.summary.merge([loss_summary],
        #                                          name="metric_summaries")

    def _train_step(self, train_batch):
        x_batch, y_batch = zip(*train_batch)
        x_batch = np.vstack(x_batch)
        y_batch = np.vstack(y_batch)
        lens = np.sum(x_batch != 0, axis=1)
        
        feed_dict = {
            self.input_x: x_batch,
            self.input_y: y_batch,
            self.input_lengths: lens,
            self.dropout_keep_prob: self.config.dropout_keep_prob
        }
        feed_dict[self.learning_rate] = self.config.learning_rate
        summary_op = tf.summary.merge([self.metric_summaries, self.grad_summaries])
        run_list = [self.global_step, self.train_op, self.loss, self.accuracy, summary_op]
        step, _, loss, accuracy, summaries = self.sess.run(run_list, feed_dict)
        print("step {}, loss {:g}, acc {:g}".format(step, loss, accuracy))
        self.train_summary_writer.add_summary(summaries, step)

    def _val_step(self, data):
        print('val step')
