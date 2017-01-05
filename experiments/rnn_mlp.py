import numpy as np
import tensorflow as tf

from datasets.dataset import DataSet
from experiments.mlp import mlp, layer, last_dim
from experiments.rnn import build_rnn, LSTM


def train(epochs=50, batch_size=100, learning_rate=.001, layers=[100, 80, 50, 30], mlp_act=tf.nn.relu,
          hidden_state_size=30, dropout_keep_prob=0.95,
          break_at_ve=0.07):
    dataset = DataSet(batch_size=batch_size, fetch_recurrent=True)

    mlp_valid, labels_valid, sentences_valid = dataset.validation()

    inp_labels, keep_prob, rnn_logits, loss, mlp_inp, mlp_logits, outp, rnn_input, train_op, out_rnn, out_mlp, mlp_weight, rnn_weight = rnn_mlp(
        dataset,
        hidden_state_size,
        layers,
        learning_rate,
        mlp_act)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()

    def run(tensors, data, dropout_keep, mlp_weight_factor=1.0, rnn_weight_factor=1.0):
        mlp_inp_data, rnn_inp_data, label_data = data
        return sess.run(tensors, {mlp_inp: mlp_inp_data, rnn_input: rnn_inp_data,
                                  inp_labels: label_data, keep_prob: dropout_keep, mlp_weight: mlp_weight_factor,
                                  rnn_weight: rnn_weight_factor})

    def err(y, y_):
        a = y != y_
        return np.sum(a) / float(y.shape[0])

    mwf, rwf = 1., 0.
    for step in range(1000000):
        if 10 <= dataset.epoch < 20:
            mwf, rwf = 0., 1.
        elif dataset.epoch >= 20:
            mwf, rwf = 1., 1.

        batch_mlp_inp_data, batch_labels, batch_sentences = dataset.next_batch()
        run(train_op, (batch_mlp_inp_data, batch_sentences, batch_labels), dropout_keep_prob, mlp_weight_factor=mwf,
            rnn_weight_factor=rwf)
        if step % 1000 == 0:
            ty_, losst = run([outp, loss], (batch_mlp_inp_data, batch_sentences, batch_labels), 1.0)
            labels_out, lossv, rnnlogit, mlplogit, labels_rnn, labels_mlp = run(
                (outp, loss, rnn_logits, mlp_logits, out_rnn, out_mlp),
                (mlp_valid, sentences_valid, labels_valid), 1.0)

            v_err = err(labels_valid, labels_out)
            ent_rnn = batch_mean_entropy(rnnlogit) / np.log(dataset.num_labels)
            ent_mlp = batch_mean_entropy(mlplogit) / np.log(dataset.num_labels)
            print(
                "s={} e={} t_err={:.3f} v_err={:.3f} losst={:.3f} lossv={:.3f} rnn_err={:.3f} mlp_err={:.3f} ent_rnn={:.3f} ent_mlp={:.3f}".format(
                    step, dataset.epoch,
                    err(ty_, batch_labels),
                    v_err,
                    losst,
                    lossv,
                    err(labels_valid, labels_rnn),
                    err(labels_valid, labels_mlp),
                    ent_rnn,
                    ent_mlp))
            if v_err < break_at_ve:
                break
        if dataset.epoch >= epochs:
            break

    saver.save(sess, 'models/rnnmlp.ckpt')
    dataset.reset()
    sess.close()


def rnn_mlp(dataset, hidden_state_size, layers, learning_rate, mlp_act):
    mlp_input_size = len(dataset.xs[0])
    keep_prob = tf.placeholder(tf.float32)
    mlp_weight = tf.placeholder(tf.float32, shape=())
    rnn_weight = tf.placeholder(tf.float32, shape=())
    inp_labels = tf.placeholder(tf.int32, shape=(None,), name='inp_labels')
    mlp_inp, hidden_layers = mlp(layers, mlp_input_size,
                                 act=mlp_act,
                                 keep_prob=keep_prob,
                                 name="dense")
    logits_mlp = layer(hidden_layers[-1], dataset.num_labels, act=tf.identity, name='logits_mlp')
    rnn_input, _, state = build_rnn(dataset.data.max_time, dataset.data.word_dim, hidden_state_size=hidden_state_size,
                                    type=LSTM)
    logits_rnn = layer(state, dataset.num_labels, act=tf.identity, name='logits_rnn')
    logits = mlp_weight * logits_mlp + rnn_weight * logits_rnn
    out_rnn = tf.argmax(logits_rnn, last_dim(logits_rnn))
    out_mlp = tf.arg_max(logits_mlp, last_dim(logits_mlp))
    outp = tf.argmax(logits, last_dim(logits), name="output")
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, inp_labels))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return inp_labels, keep_prob, logits_rnn, loss, mlp_inp, logits_mlp, outp, rnn_input, train_op, out_rnn, out_mlp, mlp_weight, rnn_weight


def batch_mean_entropy(logits):
    return np.mean([entropy(softmax(it)) for it in logits])


EPS = 1e-9


def entropy(p):
    return -np.sum(np.multiply(p + EPS, np.log(p + EPS)))


def softmax(x):
    xn = x / np.max(x)
    return np.exp(xn) / (np.sum(np.exp(xn), axis=0) + EPS)


if __name__ == '__main__':
    train()
