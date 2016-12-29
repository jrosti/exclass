import sys
import os

import tensorflow as tf
import numpy as np

from datasets.dataset import DataSet
from experiments.mlp import mlp

b = DataSet(100)
sess = tf.Session()
inp, inp_labels, outp, train_op = mlp([100, 80, 50, 30], len(b.xs[0]), b.num_labels,
                                      learning_rate=0.0,
                                      act=tf.nn.relu,
                                      dropout_prob=None)
saver = tf.train.Saver()
sess.run(tf.initialize_all_variables())

# Restore variables from disk.
model_name = 'models/model.ckpt'  # sys.argv[1]
assert os.path.isfile(model_name), "Pass model name as argument."
saver.restore(sess, model_name)
print("Model restored.")

ys_ = sess.run(outp, {inp: b.xs})

s = sum(b.ys == ys_)
print(s/len(b.ys))

xs2 = [b.data.input_vector(h) for h in b.data.docs]
ys2_ = sess.run(outp, {inp: xs2})

labels = np.array([b.data.label_of(x) for x in b.data.docs])
print(sum(ys2_ == labels)/len(ys2_))

hs = [h for h in b.data.docs[int(len(b.data.docs)*0.9):]]
ps = sess.run(outp, {inp: [b.data.input_vector(h) for h in hs]})


def sport(lbl):
    if lbl == b.num_labels - 1:
        return 'unk'
    else:
        return b.data.label_list[lbl]


for h, p in zip(hs, ps):
    print("ENNUSTE: {} -- MERKATTU: {} {}".format(sport(p), h['sport'], 'OIKEIN' if sport(p) == h['sport'] else 'VÄÄRIN'))
    print("{} {} {}".format(h['title'], h['distance'], h['duration']))
    print("http://ontrail.net/#ex/{}".format(h['_id']))
    print("--")