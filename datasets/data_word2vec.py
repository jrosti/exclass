import os
import pickle

import numpy as np

EMBEDDINGS_NPY = 'data/embeddings.npy'
WORDDICT = 'data/worddict.pickle'


def word_vec_fn():
    assert os.path.isfile(EMBEDDINGS_NPY)
    assert os.path.isfile(WORDDICT)

    embeddings = np.load(EMBEDDINGS_NPY)
    with open(WORDDICT, 'rb') as f:
        word_dict = pickle.load(f)

    def get_vector(w):
        return embeddings[word_dict[w]] if w in word_dict else embeddings[0]

    return get_vector

