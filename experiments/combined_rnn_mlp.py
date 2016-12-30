import numpy as np
import tensorflow as tf

from datasets.dataset import DataSet
from experiments.mlp import mlp, layer, last_dim
from experiments.rnn import build_rnn


def train(epochs=200, batch_size=100, learning_rate=.001, layers=[100, 80, 50, 30], mlp_act=tf.nn.relu,
          hidden_state_size=128, dropout_keep_prob=0.85,
          break_at_ve=0.1161):
    dataset = DataSet(batch_size=batch_size, fetch_recurrent=True)

    mlp_valid, labels_valid, sentences_valid = dataset.validation()

    inp_labels, keep_prob, rnn_logits, loss, mlp_inp, mlp_logits, outp, rnn_input, train_op = rnn_mlp(dataset,
                                                                                                      hidden_state_size,
                                                                                                      layers,
                                                                                                      learning_rate,
                                                                                                      mlp_act)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()

    def run(tensors, data):
        mlp_inp_data, rnn_inp_data, label_data = data
        return sess.run(tensors, {mlp_inp: mlp_inp_data, rnn_input: rnn_inp_data,
                                  inp_labels: label_data, keep_prob: dropout_keep_prob})

    def err(y, y_):
        a = y != y_
        return np.sum(a) / float(y.shape[0])

    for step in range(1000000):
        batch_mlp_inp_data, batch_labels, batch_sentences = dataset.next_batch()
        run(train_op, (batch_mlp_inp_data, batch_sentences, batch_labels))
        if step % 100 == 0:
            ty_, losst = sess.run([outp, loss], {rnn_input: batch_sentences, mlp_inp: batch_mlp_inp_data,
                                                 inp_labels: batch_labels, keep_prob: 1.0})
            valy_, lossv, rnnlogit, mlplogit = sess.run([outp, loss, rnn_logits, mlp_logits],
                                                        {rnn_input: sentences_valid, inp_labels: labels_valid,
                                                         mlp_inp: mlp_valid, keep_prob: 1.0})

            v_err = err(valy_, labels_valid)
            ent_rnn = batch_mean_entropy(rnnlogit) / np.log(dataset.num_labels)
            ent_mlp = batch_mean_entropy(mlplogit) / np.log(dataset.num_labels)
            print("s={} e={} t_err={:.3f} v_err={:.3f} losst={:.3f} lossv={:.3f} ent_rnn={:.3f} ent_mlp={:.3f}".format(
                step, dataset.epoch,
                err(ty_, batch_labels),
                v_err,
                losst,
                lossv,
                ent_rnn,
                ent_mlp))
        if dataset.epoch >= epochs or v_err < break_at_ve:
            break

    saver.save(sess, 'models/rnnmlp.ckpt')
    dataset.reset()
    #    valy_ = sess.run(outp, {inp: to_input(valrec), keep_prob: 1.0})
    #    print("{} lrs={} dr={} lr={} a={} b={}".format(err(valy_, valy), layers, dropout_prob, lr, act, batch_size))
    #    saver.save(sess, 'models/model.ckpt')
    sess.close()


def rnn_mlp(dataset, hidden_state_size, layers, learning_rate, mlp_act):
    mlp_input_size = len(dataset.xs[0])
    keep_prob = tf.placeholder(tf.float32)
    inp_labels = tf.placeholder(tf.int32, shape=(None,), name='inp_labels')
    mlp_inp, hidden_layers = mlp(layers, mlp_input_size,
                                 act=mlp_act,
                                 dropout_prob=keep_prob)
    logits_mlp = layer(hidden_layers[-1], dataset.num_labels, act=tf.identity, name='mlp_logits')
    rnn_input, _, state = build_rnn(dataset.data.max_time, dataset.data.word_dim, hidden_state_size=hidden_state_size)
    Wout = tf.get_variable('Wout', (hidden_state_size, dataset.num_labels), initializer=tf.random_normal_initializer())
    bout = tf.get_variable('bout', (dataset.num_labels,))
    logits_rnn = tf.matmul(state, Wout) + bout
    logits = logits_mlp + logits_rnn
    outp = tf.argmax(logits, last_dim(logits), name="output")
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, inp_labels))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return inp_labels, keep_prob, logits_rnn, loss, mlp_inp, logits_mlp, outp, rnn_input, train_op


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
