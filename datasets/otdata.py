import os
import pickle
from itertools import groupby
from operator import itemgetter
from random import shuffle

import numpy as np
from pymongo import MongoClient

from utils.corpus import text_to_tokens
from .data_word2vec import word_vec_fn
from .character import Character
from snowballstemmer.finnish_stemmer import FinnishStemmer


class Data(object):

    def __init__(self, num_labels):
        self.mongo = MongoClient('mongodb://localhost/ontrail')
        self.docs = []
        self.label_list = []
        self._init_data()
        self._init_labels()
        self.xs_m = None
        self.xs_s = None
        self.num_dense = None
        self.num_sparse = None
        self.label_limit = num_labels - 1
        self.max_time = 20
        self.word_dim = 128
        self.stem = FinnishStemmer().stemWord
        self.word2vec_fn = word_vec_fn()
        self.character = Character(self)

    def create_dataset(self, user_one_hot=True, month_one_hot=True, word_dot_prod=True):
        yc = np.array([self.label_of(o) for o in self.docs])
        if os.path.isfile('xs_norm.npy'):
            xs_norm = np.load('xs_norm.npy')
        else:
            dense = np.array([self._dense_features(o) for o in self.docs])
            self.num_dense = len(dense[0])
            self.xs_m = np.mean(dense, axis=0)
            self.xs_s = np.std(dense, axis=0)
            xs_norm = (dense - self.xs_m) / self.xs_s
            if user_one_hot:
                uoh = np.array([self._user_one_hot(o) for o in self.docs])
                xs_norm = np.concatenate((xs_norm, uoh), axis=1)
            if month_one_hot:
                moh = np.array([self._month_one_hot(o) for o in self.docs])
                xs_norm = np.concatenate((xs_norm, moh), axis=1)
            if word_dot_prod:
                wdp = np.array([self._avg_word_dot_prod(o) for o in self.docs])
                xs_norm = np.concatenate((xs_norm, wdp), axis=1)
            self.num_sparse = len(xs_norm[0]) - self.num_dense
            np.save('xs_norm', xs_norm)
        return xs_norm[:self.tr_mark], yc[:self.tr_mark], xs_norm[self.tr_mark:], yc[self.tr_mark:]

    def recurrent_features(self):
        print("Loading recurrent features")
        if os.path.isfile('word_feature.npy'):
            word_feats = np.load('word_feature.npy')
        else:
            word_feats = np.array([self._word_feature(o) for o in self.docs])
            np.save('word_feature', word_feats)
        return word_feats[:self.tr_mark], word_feats[self.tr_mark:]

    def input_vector(self, doc):
        if self.xs_m is None:
            dense = np.array([self._dense_features(o) for o in self.docs])
            self.num_dense = len(dense[0])
            self.xs_m = np.mean(dense, axis=0)
            self.xs_s = np.std(dense, axis=0)
        feats = self._dense_features(doc)
        feats = (feats - self.xs_m) / self.xs_s
        feats = np.concatenate((feats, self._user_one_hot(doc), self._month_one_hot(doc), self._avg_word_dot_prod(doc)),
                               axis=0)
        return feats

    def word_feature(self, doc):
        return self._word_feature(doc)

    def _word_feature(self, o):
        do_stem = True
        title_tokens = text_to_tokens(o['title'])
        title_vecs = [self.word2vec(t, stem=do_stem) for t in title_tokens][0:self.max_time - 1] if len(title_tokens) > 0 else [
            self.word2vec('no_such_token')]
        body_vecs = [self.word2vec(t, stem=do_stem) for t in text_to_tokens(o['body'])[0:max(0, self.max_time - len(title_vecs))]]
        body_title = body_vecs + title_vecs
        res = body_title + max(0, self.max_time - len(body_title)) * [np.zeros(self.word_dim)]
        assert len(res) == self.max_time
        return np.array(res)

    def _user_one_hot(self, doc):
        y = np.zeros(len(self.users), dtype=np.float32)
        u = doc['user']
        if u in self.users:
            y[self.users.index(doc['user'])] = 1.0
        return y

    def _label_of(self, one_hot_vec):
        return self.label_list[np.argmax(one_hot_vec)]

    def _init_data(self):
        fname = 'data/otdata.pickle'
        if os.path.isfile(fname):
            print("Using cached data")
            with open(fname, 'rb') as f:
                self.docs = pickle.load(f)
        else:
            otdb = self.mongo.ontrail.exercise
            projection = dict(duration=1,
                              distance=1,
                              sport=1,
                              user=1,
                              detailElevation=1,
                              detailRepeats=1,
                              avghr=1,
                              pace=1,
                              title=1,
                              body=1,
                              creationDate=1)
            self.docs = list(otdb.find({}, projection))
            shuffle(self.docs)
            with open(fname, 'wb') as f:
                pickle.dump(self.docs, f)
        self.docs = [d for d in self.docs if d['sport'] != 'Muu merkintä' and d['sport'] != 'Muu laji']
        self.users = sorted(list(set([o['user'] for o in self.docs])))
        self.count = len(self.docs)
        self.tr_mark = int(0.9 * self.count)

    def word2vec(self, word, stem=True):
        return self.word2vec_fn(self.stem(word)) if stem else self.word2vec_fn(word)

    def label_of(self, doc):
        iof = self.label_list.index(doc['sport'])
        return self.label_limit if iof >= self.label_limit else iof

    def _init_labels(self):
        for doc in self.docs:
            if doc['sport'] in self.LABEL_MAP.keys():
                doc['sport'] = self.LABEL_MAP[doc['sport']]
        labels = sorted([o['sport'] for o in self.docs])
        freqs = [(key, len(list(group))) for key, group in groupby(labels)]
        by_freqs = sorted(freqs, key=itemgetter(1), reverse=True)
        self.label_list = [k[0] for k in by_freqs]

    @staticmethod
    def _dense_features(doc):
        keys = ['distance', 'duration', 'detailElevation', 'avghr', 'pace', 'detailRepeats']
        return np.array([float(doc[key]) if key in doc and doc[key] is not None else 0.0 for key in keys])

    @staticmethod
    def _month_one_hot(doc):
        y = np.zeros(12, dtype=np.float32)
        y[doc['creationDate'].month - 1] = 1.0
        return y

    def _avg_word_dot_prod(self, doc):
        return np.array([self.average_product(label, doc) for label in self.label_list])

    def average_product(self, label, doc):
        aliases = [k.lower() for k, v in self.LABEL_MAP.items() if v == label]
        aliases.extend(label.lower())
        tokens = (text_to_tokens(doc['title']) + text_to_tokens(doc['body']) + ['no-such-token'])[:25]
        return np.average([np.dot(self.word2vec(token), self.word2vec(alias))
                          for token in tokens
                          for alias in aliases])

    LABEL_MAP = {
        'Kuntosali': 'Voimaharjoittelu',
        'Luisteluhiihto': 'Hiihto',
        'Perinteinen hiihto': 'Hiihto',
        'Jooga': 'Jumppa',
        'Maantiepyöräily': 'Pyöräily',
        'Maastopyöräily': 'Pyöräily',
        'Pumppi': 'Jumppa',
        'Vesijumppa': 'Jumppa',
        'Cyclocross': 'Pyöräily',
        'Sisäsoutu': 'Soutu'
    }
