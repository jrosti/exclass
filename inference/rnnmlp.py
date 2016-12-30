import os

import tensorflow as tf

from datasets.dataset import DataSet
from experiments.combined_rnn_mlp import rnn_mlp

layers = [100, 80, 50, 30]
mlp_act = tf.nn.relu
lstm_hidden_state_size = 30
dropout_keep_prob = 1.0

dataset = DataSet(batch_size=1, fetch_recurrent=True)

mlp_valid, labels_valid, sentences_valid = dataset.validation()

inp_labels, keep_prob, rnn_logits, loss, mlp_inp, mlp_logits, outp, rnn_input, train_op = rnn_mlp(dataset,
                                                                                                  lstm_hidden_state_size,
                                                                                                  layers,
                                                                                                  0.1,
                                                                                                  mlp_act)

sess = tf.Session()
saver = tf.train.Saver()
sess.run(tf.initialize_all_variables())

# Restore variables from disk.
model_name = 'models/rnnmlp.ckpt'  # sys.argv[1]
assert os.path.isfile(model_name), "Pass model name as argument."
saver.restore(sess, model_name)
print("Model restored.")
hs = [h for h in dataset.data.docs if h['user'] in ['KapteeniSolisluu', 'Pumppi60', 'Geoeläin', 'Peppi']]
rnni = [dataset.data.word_feature(h) for h in hs]
mlpi = [dataset.data.input_vector(h) for h in hs]
ps = sess.run(outp, {rnn_input: rnni, mlp_inp: mlpi, keep_prob: 1.0})


def sport(lbl):
    if lbl == dataset.num_labels - 1:
        return 'Muu'
    else:
        return dataset.data.label_list[lbl]


print("""
<html>
<head></head>
<body>
<table>
<th>Otsikko</th><th>Merkattu</th><th>Ennuste</th><th>Käyttäjä</th><th>Matka</th><th>Aika</th>
""")

for h, p in zip(hs, ps):
    correct = dataset.data.label_of(h) == p
    dist = "{:.1f}".format(h['distance'] / 1000.0) if 'distance' in h and h['distance'] else '-'
    dur = "{:.0f}".format(h['duration'] / 100.0 / 60.0) if 'duration' in h and h['duration'] else '-'
    print("""
<tr>
<td><a href=\"http://ontrail.net/#ex/{}\">{}</a></td>
<td><font color=\"{}\">{}</font></td>
<td><font color=\"{}\">{} {}</font></td>
<td>{}</td>
<td>{} km</td>
<td>{} min</td>
</tr>
""".format(h['_id'], h['title'],
           'green' if correct else 'red', h['sport'],
           'green' if correct else 'red', sport(p), 'O' if correct else 'V',
           h['user'],
           dist,
           dur))

print("""

</table>
</body>
</html>
""")
