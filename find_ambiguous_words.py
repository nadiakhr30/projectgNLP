from nltk.corpus import semcor
from collections import Counter

counter = Counter()

for sent in semcor.sents():
    for w in sent:
        if w.isalpha():
            counter[w.lower()] += 1

for word, count in counter.most_common(200):
    print(word, count)
