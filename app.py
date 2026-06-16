import streamlit as st
import numpy as np

from nlm_encoder import TransformerEncoder
from vectorspace import VSM

st.set_page_config(page_title="WSD SemCor-13")

st.title("Word Sense Disambiguation")
st.write("BERT Base Uncased + 1-NN (SemCor-13)")

# --------------------------------------------------
# Load model sekali saja
# --------------------------------------------------

@st.cache_resource
def load_model():

    encoder_cfg = {
        "model_name_or_path": "bert-base-uncased",
        "min_seq_len": 0,
        "max_seq_len": 512,
        "layers": [-1, -2, -3, -4],
        "layer_op": "sum",
        "subword_op": "mean"
    }

    encoder = TransformerEncoder(encoder_cfg)

    senses_vsm = VSM(
        "vectors/semcor13_test.txt",
        normalize=True
    )

    return encoder, senses_vsm


encoder, senses_vsm = load_model()

# --------------------------------------------------
# kata ambigu
# --------------------------------------------------

ambiguous_words = [
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

sentence = st.text_area(
    "Masukkan kalimat",
    height=120
)

target_word = st.selectbox(
    "Pilih kata ambigu",
    ambiguous_words
)

# --------------------------------------------------
# prediksi
# --------------------------------------------------

if st.button("Prediksi Sense"):

    if sentence.strip() == "":
        st.error("Masukkan kalimat terlebih dahulu")
        st.stop()

    tokens = sentence.lower().split()

    if target_word not in tokens:
        st.error(
            f"Kata '{target_word}' tidak ditemukan dalam kalimat"
        )
        st.stop()

    idx = tokens.index(target_word)

    try:

        inst_vecs = encoder.token_embeddings(
            [tokens]
        )[0][0]

        word_vec = inst_vecs[idx][1]

        word_vec = (
            word_vec /
            np.linalg.norm(word_vec)
        )

        preds = senses_vsm.most_similar_vec(
            word_vec,
            topn=None
        )

        preds = [
            (sense, score)
            for sense, score in preds
            if sense.split("_")[0].lower() == target_word.lower()
        ]

        if len(preds) == 0:
            st.error(
                f"Tidak ditemukan sense untuk kata '{target_word}'"
            )
            st.stop()

        best_sense, best_score = preds[0]

        st.success("Prediksi berhasil")

        st.subheader("Hasil")

        st.write(
            f"**Sense:** {best_sense}"
        )

        st.write(
            f"**Similarity:** {best_score:.4f}"
        )

    except Exception as e:
        st.error(str(e))