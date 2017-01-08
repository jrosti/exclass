import numpy as np
import tensorflow as tf

from datasets.dataset import DataSet
from nn.initializers import orthogonal_initializer

LSTM = 'LSTM'

BASIC = 'Basic'


class SimpleRNNCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, word_dim, state_size, act=tf.nn.tanh):
        self._state_size = state_size
        self.word_dim = word_dim
        self.act = act

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._state_size

    def __call__(self, inputs, state, scope=None):
        Whh = tf.get_variable('Whh', shape=(self._state_size, self._state_size), initializer=orthogonal_initializer())
        Wxh = tf.get_variable('Wxh', shape=(self.word_dim, self._state_size), initializer=orthogonal_initializer())
        b = tf.get_variable('bias', shape=(self._state_size,), initializer=tf.random_normal_initializer())
        output = self.act(tf.matmul(inputs, Wxh) + tf.matmul(state, Whh) + b)
        return output, output


def build_rnn(max_time, word_dim, hidden_state_size, type=BASIC):
    input = tf.placeholder(tf.float32, shape=(None, max_time, word_dim), name='rnn_input')
    if type == BASIC:
        outputs, state = tf.nn.dynamic_rnn(SimpleRNNCell(word_dim, hidden_state_size), inputs=input, dtype=tf.float32)
    elif type == LSTM:
        outputs, state = tf.nn.dynamic_rnn(tf.nn.rnn_cell.LSTMCell(hidden_state_size), inputs=input, dtype=tf.float32)
        state = outputs[:, -1]
    else:
        raise Exception
    return input, outputs, state


def rnn_classifier(learning_rate, num_labels, max_time, word_dim, hidden_state_size=30):
    inp, outputs, state = build_rnn(max_time, word_dim, hidden_state_size, type=LSTM)
    Wout = tf.get_variable('Wout', (hidden_state_size, num_labels), initializer=orthogonal_initializer())
    bout = tf.get_variable('bout', (num_labels,))
    logits = tf.matmul(state, Wout) + bout
    inp_labels = tf.placeholder(tf.int32, (None,))
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, inp_labels))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    outp = tf.argmax(logits, 1, name="output")
    return inp, inp_labels, loss, outp, train_op


def train(epochs=20, batch_size=200, learning_rate=.001):
    dataset = DataSet(batch_size=batch_size, fetch_recurrent=True)
    tf.reset_default_graph()

    _, labels_valid, sentences_valid = dataset.validation()

    inp, inp_labels, loss, outp, train_op = rnn_classifier(learning_rate,
                                                           dataset.num_labels,
                                                           dataset.data.max_time,
                                                           dataset.data.word_dim)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()

    def run(tensors, data):
        inp_data, label_data = data
        return sess.run(tensors, {inp: inp_data, inp_labels: label_data})

    def err(y, y_):
        a = y != y_
        return np.sum(a) / float(y.shape[0])

    for step in range(1000000):
        _, batch_labels, batch_sentences = dataset.next_batch()
        run(train_op, (batch_sentences, batch_labels))
        if step % 1000 == 0:
            ty_, losst = sess.run([outp, loss], {inp: batch_sentences, inp_labels: batch_labels})
            valy_, lossv = sess.run([outp, loss], {inp: sentences_valid, inp_labels: labels_valid})
            print("s={} e={} t_err={:.3f} v_err={:.3f} losst={:.2f} lossv={:.2f}".format(step, dataset.epoch,
                                                                                         err(ty_, batch_labels),
                                                                                         err(valy_, labels_valid),
                                                                                         losst,
                                                                                         lossv))
        if dataset.epoch >= epochs:
            break
    saver.save(sess, 'models/rnn.ckpt')
    dataset.reset()
    #    valy_ = sess.run(outp, {inp: to_input(valrec), keep_prob: 1.0})
    #    print("{} lrs={} dr={} lr={} a={} b={}".format(err(valy_, valy), layers, dropout_prob, lr, act, batch_size))
    #    saver.save(sess, 'models/model.ckpt')
    sess.close()




#    return err(valy_, valy)


if __name__ == '__main__':
    train()
