import tensorflow as tf
from keras.datasets import imdb
from tensorflow.contrib.rnn import GRUCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from tqdm import tqdm
from attention import attention


class LSTM:
    def __init__(self, sequence_length, num_classes,
                 text_vocab_size, text_embedding_size, dist_vocab_size, dist_embedding_size,
                 HIDDEN_SIZE = 150, ATTENTION_SIZE = 50, l2_reg_lambda=0.0):
        # Placeholders for input, output and dropout
        self.input_text = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_text')
        self.input_dist1 = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_dist1')
        self.input_dist2 = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_dist2')
        self.input_y = tf.placeholder(tf.float32, shape=[None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("text-embedding"):
            self.W_text = tf.Variable(tf.random_uniform([text_vocab_size, text_embedding_size], -1.0, 1.0),
                                      name="W_text")
            self.text_embedded_chars = tf.nn.embedding_lookup(self.W_text, self.input_text)
            self.text_embedded_chars_expanded = tf.expand_dims(self.text_embedded_chars, -1)
        with tf.device('/cpu:0'), tf.name_scope("dist1-embedding"):
            self.W_dist1 = tf.Variable(tf.random_uniform([dist_vocab_size, dist_embedding_size], -1.0, 1.0),
                                       name="W_dist1")
            self.dist1_embedded_chars = tf.nn.embedding_lookup(self.W_dist1, self.input_dist1)
            self.dist1_embedded_chars_expanded = tf.expand_dims(self.dist1_embedded_chars, -1)
        with tf.device('/cpu:0'), tf.name_scope("dist2-embedding"):
            self.W_dist2 = tf.Variable(tf.random_uniform([dist_vocab_size, dist_embedding_size], -1.0, 1.0),
                                       name="W_dist2")
            self.dist2_embedded_chars = tf.nn.embedding_lookup(self.W_dist2, self.input_dist2)
            self.dist2_embedded_chars_expanded = tf.expand_dims(self.dist2_embedded_chars, -1)

        self.embedded_chars_expanded = tf.concat([self.text_embedded_chars_expanded,
                                                  self.dist1_embedded_chars_expanded,
                                                  self.dist2_embedded_chars_expanded], 2)

        embedding_size = text_embedding_size + 2 * dist_embedding_size

        # (Bi-)RNN layer(-s)
        self.rnn_outputs, _ = bi_rnn(GRUCell(HIDDEN_SIZE), GRUCell(HIDDEN_SIZE),
                                     inputs=self.embedded_chars_expanded, sequence_length=sequence_length,
                                     dtype=tf.float32)
        print(self.rnn_outputs)
        tf.summary.histogram('RNN_outputs', self.rnn_outputs)

        # Attention layer
        with tf.name_scope('Attention_layer'):
            attention_output, alphas = attention(self.rnn_outputs, ATTENTION_SIZE, return_alphas=True)
            tf.summary.histogram('alphas', alphas)

        # Dropout
        self.drop = tf.nn.dropout(attention_output, self.dropout_keep_prob)

        # Fully connected layer
        with tf.name_scope('Fully_connected_layer'):
            W = tf.Variable(
                tf.truncated_normal([HIDDEN_SIZE * 2, 1], stddev=0.1))  # Hidden size is multiplied by 2 for Bi-RNN
            b = tf.Variable(tf.constant(0., shape=[1]))
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
