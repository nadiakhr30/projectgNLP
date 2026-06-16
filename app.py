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

.stApp{
    background: linear-gradient(
        135deg,
        #0f172a 0%,
        #1e293b 100%
    );
}

section[data-testid="stSidebar"]{
    background-color:#111827;
}

.hero{
    padding:30px;
    border-radius:20px;
    background:rgba(255,255,255,0.08);
    backdrop-filter:blur(10px);
    border:1px solid rgba(255,255,255,0.1);
    margin-bottom:25px;
}

.hero-title{
    font-size:42px;
    font-weight:700;
    color:white;
}

.hero-sub{
    color:#cbd5e1;
    font-size:16px;
    margin-top:10px;
}

.stat-card{
    background:white;
    border-radius:16px;
    padding:20px;
    text-align:center;
    box-shadow:0 4px 15px rgba(0,0,0,0.15);
}

.stat-title{
    font-size:14px;
    color:#64748b;
}

.stat-value{
    font-size:24px;
    font-weight:700;
    color:#2563eb;
}

.result-card{
    background:linear-gradient(
        135deg,
        #2563eb,
        #7c3aed
    );
    color:white;
    border-radius:16px;
    padding:25px;
    text-align:center;
}

.result-title{
    font-size:14px;
    opacity:0.85;
}

.result-value{
    font-size:24px;
    font-weight:700;
    margin-top:10px;
}

.stButton > button{
    width:100%;
    height:50px;
    border:none;
    border-radius:12px;
    background:linear-gradient(
        135deg,
        #2563eb,
        #7c3aed
    );
    color:white;
    font-size:16px;
    font-weight:600;
}

.stButton > button:hover{
    background:linear-gradient(
        135deg,
        #1d4ed8,
        #6d28d9
    );
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
        Sistem disambiguasi makna kata menggunakan
        BERT Base Uncased dan metode
        1-Nearest Neighbor (1-NN)
        berbasis dataset SemCor-13.
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