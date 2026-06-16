import json
import os

DATASET_DIR = "data/SemCor-13"

ambiguous_words = sorted([
    d for d in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, d))
])


def get_word_classes(word, setname=None):

    with open(
        f"{DATASET_DIR}/{word}/classes_map.txt"
    ) as f:
        return json.load(f)


def load_instances(word, split, setname=None, mode="regular"):

    instances = []

    with open(f"{DATASET_DIR}/{word}/{split}.data.txt") as df:
        for line in df:

            idx, sent = line.rstrip().split("\t", 1)

            instances.append({
                "tokens": sent.split(),
                "idx": int(idx),
                "class": None
            })

    classes = get_word_classes(word)

    with open(f"{DATASET_DIR}/{word}/{split}.gold.txt") as gf:

        for i, line in enumerate(gf):

            cls_id = line.strip()

            instances[i]["class"] = classes[cls_id]

    return instances
