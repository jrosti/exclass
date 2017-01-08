import sys
import os

import tensorflow as tf
import numpy as np

from datasets.dataset import DataSet
from nn.rnn import rnn_classifier, to_input

b = DataSet(100, fetch_recurrent=True)
sess = tf.Session()
inp, inp_labels, loss, outp, train_op = rnn_classifier(0., b.num_labels)
saver = tf.train.Saver()
sess.run(tf.initialize_all_variables())

# Restore variables from disk.
model_name = 'models/rnn.ckpt'  # sys.argv[1]
assert os.path.isfile(model_name), "Pass model name as argument."
saver.restore(sess, model_name)
print("Model restored.")

hs = [h for h in b.data.docs if h['user'] == 'Peppi']
feats = [b.data.word_feature(h) for h in hs]
ps = sess.run(outp, {inp: to_input(feats)})


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