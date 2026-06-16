import streamlit as st
import numpy as np

from nlm_encoder import TransformerEncoder
from vectorspace import VSM

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Word Sense Disambiguation",
    layout="wide"
)

# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown("""
<style>

[data-testid="stAppViewContainer"]{
    background:#f4f7fb;
}

.main .block-container{
    max-width:1200px;
    padding-top:2rem;
    padding-bottom:2rem;
}

section[data-testid="stSidebar"]{
    background:#ffffff;
    border-right:1px solid #e5e7eb;
}

.hero{
    background:white;
    padding:35px;
    border-radius:24px;
    border:1px solid #e5e7eb;
    box-shadow:0 10px 30px rgba(15,23,42,.06);
    margin-bottom:30px;
}

.hero-title{
    font-size:46px;
    font-weight:800;
    color:#0f172a;
    margin-bottom:10px;
}

.hero-sub{
    color:#64748b;
    font-size:17px;
    line-height:1.8;
}

.stat-card{
    background:white;
    border-radius:20px;
    padding:28px;
    text-align:center;
    border:1px solid #e5e7eb;
    box-shadow:0 4px 15px rgba(15,23,42,.05);
}

.stat-title{
    color:#64748b;
    font-size:14px;
    margin-bottom:10px;
}

.stat-value{
    color:#2563eb;
    font-size:30px;
    font-weight:700;
}

.result-card{
    background:white;
    border-radius:20px;
    padding:30px;
    text-align:center;
    border-left:6px solid #2563eb;
    box-shadow:0 10px 25px rgba(15,23,42,.08);
}

.result-title{
    color:#64748b;
    font-size:14px;
    text-transform:uppercase;
    letter-spacing:1px;
}

.result-value{
    color:#0f172a;
    font-size:26px;
    font-weight:700;
    margin-top:12px;
}

.stTextArea textarea{
    border-radius:14px !important;
    border:1px solid #dbe2ea !important;
}

.stSelectbox > div > div{
    border-radius:14px !important;
}

.stButton > button{
    width:100%;
    height:56px;
    border:none;
    border-radius:14px;
    background:#2563eb;
    color:white;
    font-size:16px;
    font-weight:700;
    transition:.2s;
}

.stButton > button:hover{
    background:#1d4ed8;
    transform:translateY(-1px);
}

[data-testid="stMetric"]{
    background:white;
    padding:15px;
    border-radius:16px;
}

.stProgress > div > div > div{
    background:#2563eb;
}

h1,h2,h3{
    color:#0f172a;
}

label{
    font-weight:600 !important;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# HEADER
# ==================================================

st.markdown("""
<div class="hero">
    <div class="hero-title">
        Word Sense Disambiguation
    </div>

    <div class="hero-sub">
        Sistem Word Sense Disambiguation berbasis
        BERT Base Uncased dan metode 1-Nearest Neighbor (1-NN)
        untuk menentukan makna kata ambigu berdasarkan
        konteks kalimat pada dataset SemCor-13.
    </div>
</div>
""", unsafe_allow_html=True)

# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.title("Informasi Model")

    st.markdown("""
    **Model**  
    BERT Base Uncased

    **Metode**  
    1-Nearest Neighbor (1-NN)

    **Dataset**  
    SemCor-13

    ---
    
    Aplikasi ini menentukan makna kata ambigu
    berdasarkan embedding kontekstual yang
    dihasilkan oleh BERT dan dibandingkan
    dengan sense vector hasil pelatihan
    pada dataset SemCor-13.
    """)

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
# STATISTIK
# ==================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-title">Jumlah Kata Ambigu</div>
        <div class="stat-value">13</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-title">Model</div>
        <div class="stat-value">BERT</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-title">Metode</div>
        <div class="stat-value">1-NN</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

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

st.subheader("Input Kalimat")

sentence = st.text_area(
    "Masukkan kalimat yang mengandung kata ambigu",
    height=120,
    placeholder="Contoh: The case was discussed in court yesterday."
)

target_word = st.selectbox(
    "Pilih kata ambigu",
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

        st.markdown("---")
        st.subheader("Hasil Prediksi")

        c1, c2 = st.columns(2)

        with c1:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-title">
                    Sense Terpilih
                </div>
                <div class="result-value">
                    {best_sense}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-title">
                    Similarity Score
                </div>
                <div class="result-value">
                    {best_score:.4f}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")

        st.progress(
            min(float(best_score), 1.0)
        )

        st.caption(
            "Nilai similarity menunjukkan tingkat kemiripan "
            "antara embedding kata dalam konteks kalimat "
            "dengan sense vector yang tersimpan pada model."
        )

    except Exception as e:
        st.error(str(e))