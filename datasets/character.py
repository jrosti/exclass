from functools import reduce


class Character(object):

    def __init__(self, data):
        self.data = data
        self.char_index = None

    def setup(self):
        self.char_index = sorted(list(reduce(lambda acc, d: acc | set(d['title']), self.data.docs, set())))

    def char_one_hot(self):
        pass
