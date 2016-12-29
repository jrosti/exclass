import os
import pickle

import numpy as np

from random import shuffle
from itertools import groupby
from operator import itemgetter

from pymongo import MongoClient

from utils.corpus import text_to_tokens
from . data_word2vec import word_vec_fn


class Data(object):

    def __init__(self, num_labels):
        self.mongo = MongoClient('mongodb://localhost/ontrail')
        self.docs = []
        self.label_list = []
        self._init_data()
        self._init_labels()
        self.xs_m = None
        self.xs_s = None
        self.word2vec = word_vec_fn()
        self.label_limit = num_labels - 1
        self.max_time = 10
        self.word_dim = 128

    def label_of(self, doc):
        iof = self.label_list.index(doc['sport'])
        return self.label_limit if iof >= self.label_limit else iof

    def create_dataset(self, user_one_hot=True, month_one_hot=True):
        yc = np.array([self.label_of(o) for o in self.docs])
        dense = np.array([self._dense_features(o) for o in self.docs])
        self.xs_m = np.mean(dense, axis=0)
        self.xs_s = np.std(dense, axis=0)
        xs_norm = (dense - self.xs_m) / self.xs_s
        if user_one_hot:
            uoh = np.array([self._user_one_hot(o) for o in self.docs])
            xs_norm = np.concatenate((xs_norm, uoh), axis=1)
        if month_one_hot:
            moh = np.array([self._month_one_hot(o) for o in self.docs])
            xs_norm = np.concatenate((xs_norm, moh), axis=1)
        return xs_norm[:self.tr_mark], yc[:self.tr_mark], xs_norm[self.tr_mark:], yc[self.tr_mark:]

    def recurrent_features(self):
        word_feats = [self._word_feature(o) for o in self.docs]
        return word_feats[:self.tr_mark], word_feats[self.tr_mark:]

    def input_vector(self, doc):
        assert self.xs_m is not None, "You must create dataset first"
        feats = self._dense_features(doc)
        feats = (feats - self.xs_m) / self.xs_s
        feats = np.concatenate((feats, self._user_one_hot(doc), self._month_one_hot(doc)), axis=0)
        return feats

    def word_feature(self, doc):
        return self._word_feature(doc)

    def _word_feature(self, o):
        title_vecs = [self.word2vec('no_such_token')] + \
                     [self.word2vec(t) for t in text_to_tokens(o['title'])][0:self.max_time-1]
        body_vecs = [self.word2vec(t) for t in text_to_tokens(o['body'])[0:max(0, self.max_time-len(title_vecs))]]
        body_title = body_vecs + title_vecs
        res = body_title + max(0, self.max_time - len(body_title)) * [np.zeros(self.word_dim)]
        assert len(res) == self.max_time
        return np.array(res)

    def _user_one_hot(self, doc):
        y = np.zeros(len(self.users), dtype=np.float32)
        y[self.users.index(doc['user'])] = 1.0
        return y

    def _label_of(self, one_hot_vec):
        return self.label_list[np.argmax(one_hot_vec)]

    def _init_data(self):
        fname = 'data/otdata.pickle'
        if os.path.isfile(fname):
            print("Using cached data")
            with open(fname, 'rb') as f:
                self.docs = pickle.load(f)#[0:10000]
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

        self.users = sorted(list(set([o['user'] for o in self.docs])))
        self.count = len(self.docs)
        self.tr_mark = int(0.9 * self.count)

    def _init_labels(self):
        label_map = {
            'Kuntosali': 'Voimaharjoittelu',
            'Hiihto': 'Luisteluhiihto',
            'Jooga': 'Jumppa',
            'Maantiepyöräily': 'Pyöräily',
            'Maastopyöräily': 'Pyöräily',
            'Pumppi': 'Jumppa',
            'Vesijumppa': 'Jumppa',
            'Cyclocross': 'Pyöräily',

        }
        for doc in self.docs:
            if doc['sport'] in label_map.keys():
                doc['sport'] = label_map[doc['sport']]
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


