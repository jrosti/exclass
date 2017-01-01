from datasets.otdata import Data


class DataSet(object):

    def __init__(self, batch_size, fetch_recurrent=False):
        self.num_labels = 18
        self.data = Data(self.num_labels)
        self.xs, self.ys, self.xc, self.yc = self.data.create_dataset(user_one_hot=True,
                                                                      month_one_hot=True)
        self.current_batch = -1
        self.epoch = 0
        self.batch_size = batch_size
        self.fetch_recurrent = fetch_recurrent
        self.num_dense = self.data.num_dense
        if fetch_recurrent:
            self.s_train, self.s_valid = self.data.recurrent_features()
        else:
            self.s_train, self.s_valid = (None, None)

    def reset(self):
        self.current_batch = -1
        self.epoch = 0

    def next_batch(self):
        self.current_batch += 1
        if (self.batch_size + 1) * self.current_batch >= len(self.xs):
            self.current_batch = 0
            self.epoch += 1
        xb, yb = (self.xs[self.batch_size * self.current_batch: self.batch_size * (self.current_batch + 1)],
                  self.ys[self.batch_size * self.current_batch: self.batch_size * (self.current_batch + 1)])
        if not self.fetch_recurrent:
            return xb, yb
        else:
            return xb, yb, \
                   self.s_train[self.batch_size * self.current_batch: self.batch_size * (self.current_batch + 1)]

    def validation(self):
        return self.xc, self.yc, self.s_valid
