import snowballstemmer.finnish_stemmer as stemmer

s = stemmer.FinnishStemmer()

with open('corpus.txt', 'r') as r, open('corpus-stemmed.txt', 'w') as o:
    for line in r.readlines():
        o.write(s.stemWord(line.strip()))
        o.write('\n')
