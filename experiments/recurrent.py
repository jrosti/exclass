import numpy as np
import tensorflow as tf

from datasets.dataset import DataSet

MAX_TIME = 14


def orthogonal_initializer(scale=0.95):
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        # pick the one with the correct shape
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)  # this needs to be corrected to float32
        return tf.constant(scale * q[:shape[0], :shape[1]], dtype=tf.float32)

    return _initializer


class XRNNCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, act=tf.nn.tanh):
        self._state_size = 100
        self.act = act

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._state_size

    def __call__(self, inputs, state, scope=None):
        Whh = tf.get_variable('Whh', shape=(self._state_size, self._state_size), initializer=orthogonal_initializer())
        Wxh = tf.get_variable('Wxh', shape=(128, self._state_size))
        output = self.act(tf.matmul(inputs, Wxh) + tf.matmul(state, Whh))
        return output, output


def build_rnn():
    input = tf.placeholder(tf.float32, shape=(None, MAX_TIME, 128), name='input')
    outputs, state = tf.nn.dynamic_rnn(tf.nn.rnn_cell.LSTMCell(100), inputs=input, dtype=tf.float32)
    return input, outputs, state


def train(epochs=2, batch_size=20, learning_rate=.001):
    batches = DataSet(batch_size=batch_size, fetch_recurrent=True)
    tf.reset_default_graph()

    _, valy, valrec = batches.validation()

    inp, outputs, state = build_rnn()
    Wout = tf.get_variable('Wout', (100, 18), initializer=orthogonal_initializer())
    bout = tf.get_variable('bout', (18,))
    logits = tf.nn.relu(tf.matmul(outputs[:, -1], Wout) + bout)
    inp_labels = tf.placeholder(tf.int32, (None,))
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, inp_labels))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    outp = tf.argmax(logits, 1, name="output")

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()

    def pad_zeros(inp):
        zd = np.zeros((MAX_TIME - len(inp), 128))
        res = np.concatenate((inp, zd))
        return res

    def to_input(sb):
        batch = np.array([pad_zeros(np.array(s[0][0:14])) for s in sb])
        return batch

    def run(tensors, data):
        inp_data, label_data = data
        return sess.run(tensors, {inp: inp_data, inp_labels: label_data})

    def err(y, y_):
        a = y != y_
        return np.sum(a) / float(y.shape[0])

    for step in range(1000000):
        _, yb, sb = batches.next_batch()
        run(train_op, (to_input(sb), yb))
        if step % 100 == 0:
            ty_, losst = sess.run([outp, loss], {inp: to_input(sb), inp_labels: yb})
            valy_, lossv = sess.run([outp, loss], {inp: to_input(valrec), inp_labels: valy})
            print("s={} e={} t_err={:.3f} v_err={:.3f} losst={:.2f} lossv={:.2f}".format(step, batches.epoch,
                                                                                         err(ty_, yb),
                                                                                         err(valy_, valy),
                                                                                         losst,
                                                                                         lossv))
        if batches.epoch >= epochs:
            break
    batches.reset()
    #    valy_ = sess.run(outp, {inp: to_input(valrec), keep_prob: 1.0})
    #    print("{} lrs={} dr={} lr={} a={} b={}".format(err(valy_, valy), layers, dropout_prob, lr, act, batch_size))
    #    saver.save(sess, 'models/model.ckpt')
    sess.close()


#    return err(valy_, valy)


if __name__ == '__main__':
    train()
