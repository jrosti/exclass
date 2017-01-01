import re
import operator


from pymongo import MongoClient

tags_re = re.compile(r'<.*?>')
entities_re = re.compile(r'&.*?;')


def remove_tags(text):
    text = re.sub(entities_re, ' ', re.sub(tags_re, ' ', text))
    return text


def text_to_tokens(s):
    raw_text = remove_tags(s)
    lower_text = raw_text.lower()
    text_chars = re.sub(r"[^a-zåäæö0-9]+", " ", lower_text)
    tokens = text_chars.split(" ")
    return [t for t in tokens if len(t) > 0]


def extract_corpus():
    corpus = []
    mongo = MongoClient('mongodb://localhost/ontrail')

    otdb = mongo.ontrail.exercise
    projection = dict(title=1,
                      body=1,
                      sport=1,
                      user=1, comments=1)
    docs = list(otdb.find({}, projection))
    assert len(docs) > 190000
    sorted_docs = sorted(docs, key=operator.itemgetter('sport'))
    for doc in sorted_docs:
        corpus.extend(text_to_tokens(doc['sport']))
        corpus.extend(text_to_tokens(doc['title']))
        corpus.extend(text_to_tokens(doc['user']))
        corpus.extend(text_to_tokens(doc['body']))
        if 'comments' in doc:
            for c in doc['comments']:
                if 'body' in c and c['body']:
                    corpus.extend(text_to_tokens(c['body']))
    assert len(corpus) > 300000
    with open('corpus.txt', 'w') as f:
        for w in corpus:
            f.write(w + '\n')
    return corpus


def load():
    print("Loading corpus")
    with open('corpus.txt', 'r') as f:
        return f.read().splitlines()


if __name__ == '__main__':
    extract_corpus()