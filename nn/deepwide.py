import tensorflow as tf
import numpy as np

from nn.mlp import mlp, layer, last_dim
from datasets.dataset import DataSet

DENSE_DEFAULT = {
    'layers': [50, 50, 50, 50, 50],
    'act': tf.nn.tanh
}

SPARSE_DEFAULT = {
    'layers': [18],
    'act': tf.nn.softmax
}


def deep_wide(dense_inp_width, sparse_inp_width, output_width, dense=DENSE_DEFAULT, sparse=SPARSE_DEFAULT, learning_rate=.001):
    with tf.variable_scope('deep_and_wide'):
        keep_prob = tf.placeholder(tf.float32)
        with tf.variable_scope('dense'):
            dense_in, h = mlp(dense['layers'], dense_inp_width, act=dense['act'], keep_prob=keep_prob)
            dense_out = h[-1]
        with tf.variable_scope('sparse'):
            sparse_in, h = mlp(sparse['layers'], sparse_inp_width, act=sparse['act'], keep_prob=keep_prob)
            sparse_out = h[-1]
        with tf.variable_scope('merge'):
            concat = tf.concat(1, (sparse_out, dense_out))
            logits = layer(concat, output_width, tf.identity)
            outp = tf.argmax(logits, last_dim(logits), name="output")
        inp_labels = tf.placeholder(tf.int32, shape=(None,), name='inp_labels')
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, inp_labels))
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return locals()


def train(batch_size=100, dropout_keep_prob=1.0, epochs=20):

    d = DataSet(batch_size)
    x_valid, labels_valid, _ = d.validation()

    dw = deep_wide(d.data.num_dense, d.data.num_sparse, d.num_labels)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()

    def run(tensors, data, keep_prob):
        inp_data, label_data = data
        return sess.run(tensors, {dw['dense_in']: inp_data[:, :d.num_dense],
                                  dw['sparse_in']: inp_data[:, d.num_dense:],
                                  dw['inp_labels']: label_data,
                                  dw['keep_prob']: keep_prob})

    def err(y, y_):
        a = y != y_
        return np.sum(a) / float(y.shape[0])

    outp, loss, keep_prob, train_op = [dw[o] for o in ['outp', 'loss', 'keep_prob', 'train_op']]
    for step in range(1000000):
        x_batch, labels_batch = d.next_batch()
        run(train_op, (x_batch, labels_batch), dropout_keep_prob)
        if step % 1000 == 0:
            labels_out_train, loss_train = run((outp, loss), (x_batch, labels_batch), 1.0)
            labels_out_valid, loss_valid = run([outp, loss], (x_valid, labels_valid), 1.0)
            print("s={} e={} t_err={:.3f} v_err={:.3f} losst={:.3f}, lossv={:.3f}".format(step, d.epoch,
                                                                                          err(labels_out_train, labels_batch),
                                                                                          err(labels_out_valid, labels_valid),
                                                                                          loss_train,
                                                                                          loss_valid))
            if d.epoch >= epochs:
                break

if __name__ == '__main__':
    train()