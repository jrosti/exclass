from datasets.otdata import Data

import random

import numpy as np


class DataSet(object):

    def __init__(self, batch_size, fetch_recurrent=False):
        self.num_labels = 18
        self.data = Data(self.num_labels)
        self.xs, self.ys, self.xc, self.yc = self.data.create_dataset(user_one_hot=True,
                                                                      month_one_hot=True)
        self.s_train, self.s_valid = self.data.recurrent_features()

        self.current_batch = -1
        self.epoch = 0
        self.batch_size = batch_size
        self.fetch_recurrent = fetch_recurrent
        self.num_dense = self.data.num_dense
        label_to_indices = {}
        for i, label in enumerate(self.ys):
            if label in label_to_indices:
                label_to_indices[label].append(i)
            else:
                label_to_indices[label] = [i]
        self.label_to_indices = label_to_indices

    def reset(self):
        self.current_batch = -1
        self.epoch = 0

    def next_batch(self):
        self.current_batch += 1
        batch = []
        for label in range(self.num_labels):
            batch.extend([random.choice(self.label_to_indices[label]) for _ in range(self.batch_size//self.num_labels)])
        b = np.array(batch, dtype=np.int32)
        return self.xs[b], self.ys[b], self.s_train[b]

    def validation(self):
        return self.xc, self.yc, self.s_valid

