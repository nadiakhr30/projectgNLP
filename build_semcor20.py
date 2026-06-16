import os
import json
import random
from collections import defaultdict, Counter

from nltk.corpus import semcor
from nltk.tree import Tree
from sklearn.model_selection import train_test_split

TARGET_WORDS = [
    "case",
    "face",
    "form",
    "head",
    "interest",
    "life",
    "light",
    "matter",
    "point",
    "state",
    "time",
    "way",
    "work"
]

MIN_INSTANCES = 20
MAX_PER_SENSE = 100
TEST_SIZE = 0.2
RANDOM_STATE = 42

OUTPUT_DIR = "data/SemCor-13"

# ==========================================
# PASS 1 : hitung semua sense
# ==========================================

sense_counts = defaultdict(Counter)

for sent in semcor.tagged_sents(tag="sem"):
    for item in sent:

        if not isinstance(item, Tree):
            continue

        label = item.label()

        if not hasattr(label, "synset"):
            continue

        lemma = label.name().split(".")[0]

        if lemma not in TARGET_WORDS:
            continue

        sense = label.synset().name()

        sense_counts[lemma][sense] += 1


valid_senses = {}

for word in TARGET_WORDS:

    senses = {
        s: c
        for s, c in sense_counts[word].items()
        if c >= MIN_INSTANCES
    }

    if len(senses) >= 2:
        valid_senses[word] = list(senses.keys())

# ==========================================
# PASS 2 : kumpulkan instance
# ==========================================

instances = defaultdict(list)

for sent in semcor.tagged_sents(tag="sem"):

    tokens = []

    entries = []

    for item in sent:

        if isinstance(item, Tree):

            words = item.leaves()

            start = len(tokens)

            tokens.extend(words)

            label = item.label()

            if hasattr(label, "synset"):

                lemma = label.name().split(".")[0]

                if lemma in valid_senses:

                    sense = label.synset().name()

                    if sense in valid_senses[lemma]:

                        entries.append(
                            (lemma, sense, start)
                        )

        else:
            tokens.extend(item)

    for lemma, sense, idx in entries:

        instances[lemma].append({
            "tokens": tokens,
            "idx": idx,
            "sense": sense
        })

# ==========================================
# BUILD DATASET
# ==========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

for word, data in instances.items():

    word_dir = os.path.join(OUTPUT_DIR, word)

    os.makedirs(word_dir, exist_ok=True)

    sense_groups = defaultdict(list)

    for inst in data:
        sense_groups[inst["sense"]].append(inst)

    filtered = []

    sense_names = []

    for sense, items in sense_groups.items():

        random.shuffle(items)

        items = items[:MAX_PER_SENSE]

        filtered.extend(items)

        sense_names.append(sense)

    classes_map = {
        str(i): sense
        for i, sense in enumerate(sorted(sense_names))
    }

    sense_to_id = {
        sense: idx
        for idx, sense in enumerate(sorted(sense_names))
    }

    train, test = train_test_split(
        filtered,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=[x["sense"] for x in filtered]
    )

    with open(
        os.path.join(word_dir, "classes_map.txt"),
        "w"
    ) as f:
        json.dump(classes_map, f, indent=2)

    def save_split(split_data, split_name):

        data_path = os.path.join(
            word_dir,
            f"{split_name}.data.txt"
        )

        gold_path = os.path.join(
            word_dir,
            f"{split_name}.gold.txt"
        )

        with open(data_path, "w") as df, \
             open(gold_path, "w") as gf:

            for inst in split_data:

                df.write(
                    f"{inst['idx']}\t{' '.join(inst['tokens'])}\n"
                )

                gf.write(
                    f"{sense_to_id[inst['sense']]}\n"
                )

    save_split(train, "train")
    save_split(test, "test")

    print(
        word,
        len(train),
        len(test),
        len(classes_map)
    )

print("\nDONE")