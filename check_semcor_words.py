from nltk.corpus import semcor
from collections import Counter

words = [
    "bat","cell","charge","draft","file",
    "jam","light","match","mint","mouse",
    "organ","paper","pole","port","ring",
    "rock","scale","star","terminal","watch"
]

counter = Counter()

for sent in semcor.sents():
    for w in sent:
        w = w.lower()
        if w in words:
            counter[w] += 1

for w,c in sorted(counter.items()):
    print(w, c)