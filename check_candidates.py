from nltk.corpus import semcor
from collections import Counter

WORDS = [
    "time","state","way","point","case",
    "line","field","interest","power","matter",
    "right","order","face","head","course",
    "party","work","life","change","service",
    "force","form","position","subject","issue",
    "record","level","figure","charge","light"
]

counter = Counter()

for sent in semcor.sents():
    for token in sent:
        token = token.lower()
        if token in WORDS:
            counter[token] += 1

for w,c in sorted(counter.items(), key=lambda x:x[1], reverse=True):
    print(w, c)
