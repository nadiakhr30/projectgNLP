import streamlit as st
import numpy as np

from nlm_encoder import TransformerEncoder
from vectorspace import VSM

st.set_page_config(
    page_title="Word Sense Disambiguation",
    layout="centered"
)

# ==================================================
# STYLE
# ==================================================

st.markdown("""
<style>
.main {
    padding-top: 1rem;
}

.result-box {
    padding: 1rem;
    border-radius: 10px;
    border: 1px solid #dddddd;
    background-color: #f8f9fa;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# HEADER
# ==================================================

st.title("Word Sense Disambiguation")

st.markdown("""
Sistem disambiguasi makna kata menggunakan **BERT Base Uncased**
dan metode **1-Nearest Neighbor (1-NN)** berdasarkan dataset
**SemCor-13**.

Masukkan kalimat yang mengandung kata ambigu,
kemudian pilih kata target untuk memperoleh prediksi sense.
""")

# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.header("Informasi Model")

    st.write("Model : BERT Base Uncased")
    st.write("Metode : 1-NN")
    st.write("Dataset : SemCor-13")

    st.divider()

    st.write(
        """
        Aplikasi ini menentukan makna kata ambigu
        berdasarkan kemiripan embedding kontekstual
        dengan sense vector yang telah dibangun
        dari dataset SemCor.
        """
    )

# ==================================================
# LOAD MODEL
# ==================================================

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

# ==================================================
# KATA AMBIGU
# ==================================================

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

st.subheader("Input")

sentence = st.text_area(
    "Kalimat",
    height=120,
    placeholder="Contoh: The case was discussed in court yesterday."
)

target_word = st.selectbox(
    "Kata ambigu",
    ambiguous_words
)

# ==================================================
# PREDIKSI
# ==================================================

if st.button(
    "Prediksi Sense",
    use_container_width=True
):

    if sentence.strip() == "":
        st.warning("Masukkan kalimat terlebih dahulu.")
        st.stop()

    tokens = sentence.lower().split()

    if target_word not in tokens:
        st.warning(
            f"Kata '{target_word}' tidak ditemukan dalam kalimat."
        )
        st.stop()

    idx = tokens.index(target_word)

    try:

        with st.spinner("Memproses prediksi..."):

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
                if sense.lower().startswith(
                    target_word.lower() + "."
                )
            ]

            if len(preds) == 0:
                st.error(
                    f"Tidak ditemukan sense untuk kata '{target_word}'"
                )
                st.stop()

            best_sense, best_score = preds[0]

        st.subheader("Hasil Prediksi")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Sense",
                best_sense
            )

        with col2:
            st.metric(
                "Similarity",
                f"{best_score:.4f}"
            )

        st.progress(
            min(float(best_score), 1.0)
        )

        st.caption(
            "Nilai similarity menunjukkan tingkat kemiripan "
            "antara embedding kata dalam kalimat dengan "
            "sense vector yang tersimpan pada model."
        )

    except Exception as e:
        st.error(str(e))