from itertools import accumulate

import numpy as np
import tensorflow as tf

from datasets.dataset import DataSet
from experiments.mlp import mlp


def train(epochs=2, layers=[30, 30], lr=0.015, act=tf.nn.relu, batch_size=700, dropout_prob=0.9):
    batches = DataSet(batch_size=batch_size, fetch_recurrent=True)
    tf.reset_default_graph()
    keep_prob = tf.placeholder(tf.float32)
    inp, inp_labels, outp, train_op = mlp(layers, 128, batches.num_labels,
                                          learning_rate=lr,
                                          act=act,
                                          dropout_prob=keep_prob)
    _, valy, valrec = batches.validation()
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()

    def to_input(sb):
        batch = np.array([np.average(s[0], axis=0) for s in sb])
        return batch

    def run(tensors, data):
        inp_data, label_data = data
        return sess.run(tensors, {inp: inp_data, inp_labels: label_data, keep_prob: dropout_prob})

    def err(y, y_):
        a = y != y_
        return np.sum(a) / float(y.shape[0])

    for step in range(1000000):
        _, yb, sb = batches.next_batch()
        run(train_op, (to_input(sb), yb))
        if step % 100 == 0:
            ty_ = sess.run(outp, {inp: to_input(sb), keep_prob: 1.0})
            valy_ = sess.run(outp, {inp: to_input(valrec), keep_prob: 1.0})
            print("s={} e={} t_err={:.3f} v_err={:.3f}".format(step, batches.epoch,
                                                               err(ty_, yb),
                                                               err(valy_, valy),))
        if batches.epoch >= epochs:
            break
    batches.reset()
    valy_ = sess.run(outp, {inp: to_input(valrec), keep_prob: 1.0})
    print("{} lrs={} dr={} lr={} a={} b={}".format(err(valy_, valy), layers, dropout_prob, lr, act, batch_size))
    saver.save(sess, 'models/model.ckpt')
    sess.close()
    return err(valy_, valy)


if __name__ == '__main__':
    train(epochs=30, layers=[41, 41, 41], lr=0.009, act=tf.nn.relu, dropout_prob=.8)
