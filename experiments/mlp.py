from itertools import accumulate

import numpy as np
import tensorflow as tf

from datasets.dataset import DataSet
from experiments.initializers import orthogonal_initializer


def shape(graph_tensor):
    return tuple(dim.value for dim in graph_tensor.get_shape())


def width(graph_tensor):
    return shape(graph_tensor)[-1]


def last_dim(graph_tensor):
    return len(shape(graph_tensor)) - 1


def stack_layers(bottom, layer_params, builder_func):
    return list(accumulate([bottom] + layer_params, builder_func))


i = 0


def layer(inp, outp_width, act=tf.nn.relu, dropout=None, name=None):
    global i
    i += 1
    W = tf.get_variable('W' + str(i), shape=(width(inp), outp_width), dtype=tf.float32,
                        initializer=orthogonal_initializer())
    b = tf.Variable(tf.random_normal([outp_width], stddev=0.35), 'b')
    pre_act = tf.matmul(inp, W) + b
    outp = act(pre_act, name=name)
    return tf.nn.dropout(outp, dropout) if dropout is not None else outp


def mlp(hidden_layer_widths, input_width, act=tf.nn.relu, dropout_prob=None):
    inp = tf.placeholder(tf.float32, shape=(None, input_width), name="mlp_input")
    layerf = lambda input, output_width: layer(input, output_width, act, dropout_prob)
    hidden_layers = stack_layers(inp, hidden_layer_widths, layerf)
    return inp, hidden_layers


def train(epochs=None, layers=None, learning_rate=0.001, act=tf.nn.relu, batch_size=700, dropout_prob=0.9):
    batches = DataSet(batch_size=batch_size)
    tf.reset_default_graph()
    input_size = len(batches.xs[0])
    keep_prob = tf.placeholder(tf.float32)
    inp_labels = tf.placeholder(tf.int32, shape=(None,), name='mlp_inp_labels')
    inp, hidden_layers = mlp(layers, input_size,
                             act=act,
                             dropout_prob=keep_prob)

    logits = layer(hidden_layers[-1], batches.num_labels, act=tf.identity, name='logits')
    outp = tf.argmax(logits, last_dim(logits), name="output")

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, inp_labels))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    valx, valy, _ = batches.validation()
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()

    def run(tensors, data):
        inp_data, label_data = data
        return sess.run(tensors, {inp: inp_data, inp_labels: label_data, keep_prob: dropout_prob})

    def err(y, y_):
        a = y != y_
        return np.sum(a) / float(y.shape[0])

    for step in range(1000000):
        xb, yb = batches.next_batch()
        run(train_op, (xb, yb))
        if step % 100 == 0:
            ty_, losst = sess.run([outp, loss], {inp: xb, keep_prob: 1.0, inp_labels: yb})
            valy_, lossv = sess.run([outp, loss], {inp: valx, keep_prob: 1.0, inp_labels: valy})
            print("s={} e={} t_err={:.3f} v_err={:.3f} losst={:.3f}, lossv={:.3f}".format(step, batches.epoch,
                                                                                          err(ty_, yb),
                                                                                          err(valy_, valy), losst,
                                                                                          lossv))
        if batches.epoch >= epochs:
            break
    batches.reset()
    valy_ = sess.run(outp, {inp: valx, keep_prob: 1.0})
    print("{} lrs={} dr={} lr={} a={} b={}".format(err(valy_, valy), layers, dropout_prob, learning_rate, act,
                                                   batch_size))
    saver.save(sess, 'models/model.ckpt')
    sess.close()
    return err(valy_, valy)


if __name__ == '__main__':
    train(epochs=300, layers=[100, 80, 50, 30], learning_rate=0.001, act=tf.nn.relu, dropout_prob=0.99)
