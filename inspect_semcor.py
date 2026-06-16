from nltk.corpus import semcor

for sent in semcor.tagged_sents(tag='sem'):
    for item in sent:
        if hasattr(item, "label"):
            print(item)
            print(item.label())
            exit()
