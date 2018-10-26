import tensorflow as tf
from attention import attention


class AttLSTM:
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size,
                 hidden_size, l2_reg_lambda=0.0):
        # Placeholders for input, output and dropout
        self.input_text = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_text')
        self.input_y = tf.placeholder(tf.float32, shape=[None, num_classes], name='input_y')
        self.emb_dropout_keep_prob = tf.placeholder(tf.float32, name='emb_dropout_keep_prob')
        self.rnn_dropout_keep_prob = tf.placeholder(tf.float32, name='rnn_dropout_keep_prob')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        initializer = tf.keras.initializers.glorot_normal

        # Word Embedding Layer
        with tf.device('/cpu:0'), tf.variable_scope("word-embeddings"):
            self.W_text = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -0.25, 0.25), name="W_text")
            self.embedded_chars = tf.nn.embedding_lookup(self.W_text, self.input_text)

        # Dropout for Word Embedding
        with tf.variable_scope('dropout-embeddings'):
            self.embedded_chars = tf.nn.dropout(self.embedded_chars, self.emb_dropout_keep_prob)

        # Bidirectional LSTM
        with tf.variable_scope("bi-lstm"):
            _fw_cell = tf.nn.rnn_cell.LSTMCell(hidden_size, initializer=initializer())
            fw_cell = tf.nn.rnn_cell.DropoutWrapper(_fw_cell, self.rnn_dropout_keep_prob)
            _bw_cell = tf.nn.rnn_cell.LSTMCell(hidden_size, initializer=initializer())
            bw_cell = tf.nn.rnn_cell.DropoutWrapper(_bw_cell, self.rnn_dropout_keep_prob)
            self.rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                                                                  cell_bw=bw_cell,
                                                                  inputs=self.embedded_chars,
                                                                  sequence_length=self._length(self.input_text),
                                                                  dtype=tf.float32)
            self.rnn_outputs = tf.add(self.rnn_outputs[0], self.rnn_outputs[1])

        # Attention
        with tf.variable_scope('attention'):
            self.attn, self.alphas = attention(self.rnn_outputs)

        # Dropout
        with tf.variable_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.attn, self.dropout_keep_prob)

        # Fully connected layer
        with tf.variable_scope('output'):
            self.logits = tf.layers.dense(self.h_drop, num_classes, kernel_initializer=initializer())
            self.predictions = tf.argmax(self.logits, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.variable_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.input_y)
            self.l2 = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * self.l2

        # Accuracy
        with tf.variable_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")

    # Length of the sequence data
    @staticmethod
    def _length(seq):
        relevant = tf.sign(tf.abs(seq))
        length = tf.reduce_sum(relevant, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length
