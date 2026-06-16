import streamlit as st
import numpy as np
from nlm_encoder import TransformerEncoder
from vectorspace import VSM

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Word Sense Disambiguation",
    page_icon="⚙️",  # masih bisa pakai emoji untuk tab browser, optional
    layout="wide"
)

# ==================================================
# CUSTOM CSS (TANPA EMOJI, FONT INTER)
# ==================================================
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,300;14..32,400;14..32,500;14..32,600;14..32,700;14..32,800&display=swap" rel="stylesheet">
    <style>
        * {
            font-family: 'Inter', sans-serif;
        }

        [data-testid="stAppViewContainer"] {
            background: #f8fafc;
        }

        .main .block-container {
            max-width: 1200px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        section[data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid #e9edf2;
        }

        /* Hero tanpa emoji */
        .hero {
            background: white;
            border-radius: 28px;
            padding: 2rem 2.5rem;
            margin-bottom: 2rem;
            border: 1px solid #eef2f6;
            box-shadow: 0 12px 30px rgba(0,0,0,0.03);
        }

        .hero-label {
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.5px;
            color: #3b82f6;
            text-transform: uppercase;
            margin-bottom: 0.75rem;
            border-left: 3px solid #3b82f6;
            padding-left: 12px;
        }

        .hero-title {
            font-size: 2.8rem;
            font-weight: 800;
            color: #0a2540;
            letter-spacing: -0.02em;
            margin-bottom: 1rem;
        }

        .hero-sub {
            font-size: 1rem;
            line-height: 1.5;
            color: #425466;
            max-width: 90%;
            margin-bottom: 1.2rem;
        }

        .hero-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 0.8rem;
            margin-top: 0.5rem;
        }

        .tag {
            background: #f1f5f9;
            padding: 0.25rem 1rem;
            border-radius: 30px;
            font-size: 0.8rem;
            font-weight: 500;
            color: #1e293b;
            letter-spacing: 0.2px;
        }

        /* Stat card */
        .stat-card {
            background: white;
            border-radius: 24px;
            padding: 1.5rem;
            text-align: center;
            border: 1px solid #eef2f6;
            transition: all 0.2s ease;
        }
        .stat-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 24px -12px rgba(0,0,0,0.08);
            border-color: #cbd5e1;
        }
        .stat-value {
            font-size: 2.4rem;
            font-weight: 800;
            color: #0f172a;
            line-height: 1.2;
        }
        .stat-label {
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #64748b;
            margin-top: 0.5rem;
        }
        .stat-desc {
            font-size: 0.7rem;
            color: #94a3b8;
            margin-top: 0.25rem;
        }

        /* Result card */
        .result-card {
            background: white;
            border-radius: 24px;
            padding: 1.8rem;
            text-align: center;
            border-left: 5px solid #3b82f6;
            box-shadow: 0 8px 20px rgba(0,0,0,0.04);
        }
        .result-label {
            font-size: 0.7rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #64748b;
        }
        .result-value {
            font-size: 1.6rem;
            font-weight: 800;
            color: #0f172a;
            margin-top: 0.5rem;
            word-break: break-word;
        }

        /* Form styling */
        .stTextArea textarea, .stSelectbox > div > div {
            border-radius: 16px !important;
            border: 1px solid #e2e8f0 !important;
            font-size: 0.9rem;
        }

        .stButton > button {
            width: 100%;
            height: 52px;
            background: #1e293b;
            color: white;
            font-weight: 600;
            border: none;
            border-radius: 16px;
            transition: 0.2s;
            font-size: 0.95rem;
        }
        .stButton > button:hover {
            background: #0f172a;
            transform: translateY(-1px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.1);
        }

        .stProgress > div > div > div {
            background: #3b82f6;
            border-radius: 20px;
        }

        .footer {
            text-align: center;
            margin-top: 3rem;
            padding-top: 1rem;
            font-size: 0.7rem;
            color: #94a3b8;
            border-top: 1px solid #eef2f6;
        }

        h1, h2, h3, h4 {
            color: #0a2540;
            font-weight: 600;
        }

        hr {
            margin: 1.5rem 0;
        }

        .sidebar-heading {
            font-weight: 700;
            font-size: 1rem;
            letter-spacing: -0.2px;
            margin-bottom: 1rem;
            color: #0f172a;
        }

        .info-block {
            font-size: 0.85rem;
            line-height: 1.5;
            color: #334155;
        }
    </style>
""", unsafe_allow_html=True)

# ==================================================
# HEADER (TANPA EMOJI)
# ==================================================
st.markdown("""
<div class="hero">
    <div class="hero-label">WORD SENSE DISAMBIGUATION</div>
    <div class="hero-title">Menentukan Makna Kata Ambigu<br>dalam Konteks Kalimat</div>
    <div class="hero-sub">
        Sistem berbasis BERT Base Uncased dan metode 1-Nearest Neighbor (1-NN)
        untuk memetakan kata ambigu ke sense yang paling sesuai berdasarkan 
        embedding kontekstual dari dataset SemCor-13.
    </div>
    <div class="hero-tags">
        <span class="tag">BERT Base Uncased</span>
        <span class="tag">1-NN Classifier</span>
        <span class="tag">SemCor-13</span>
        <span class="tag">Contextual Embedding</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ==================================================
# SIDEBAR (TANPA EMOJI)
# ==================================================
with st.sidebar:
    st.markdown('<div class="sidebar-heading">◆ Informasi Model</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-block">
    <strong>Model</strong><br>
    <code>BERT Base Uncased</code><br><br>
    <strong>Metode</strong><br>
    <code>1-Nearest Neighbor (1-NN)</code><br><br>
    <strong>Dataset</strong><br>
    <code>SemCor-13</code><br><br>
    <hr>
    <strong>Prinsip kerja</strong><br>
    Embedding kata ambigu dibandingkan dengan <i>sense vector</i> hasil pelatihan. 
    Sense dengan cosine similarity tertinggi dipilih sebagai makna.
    <br><br>
    <strong>Contoh kalimat</strong><br>
    • The case was presented to the judge.<br>
    • She wore a watch on her wrist.<br>
    • Time flies like an arrow.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption("© 2025 • WSD System")

# ==================================================
# LOAD MODEL (CACHE)
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
    senses_vsm = VSM("vectors/semcor13_test.txt", normalize=True)
    return encoder, senses_vsm

encoder, senses_vsm = load_model()

# ==================================================
# STATISTIK (TANPA EMOJI)
# ==================================================
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-value">13</div>
        <div class="stat-label">Kata Ambigu</div>
        <div class="stat-desc">dalam SemCor-13</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-value">BERT</div>
        <div class="stat-label">Encoder</div>
        <div class="stat-desc">Base Uncased</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-value">1-NN</div>
        <div class="stat-label">Metode</div>
        <div class="stat-desc">Similarity-based</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==================================================
# INPUT KALIMAT
# ==================================================
ambiguous_words = [
    "case", "face", "form", "head", "interest", "life", "light",
    "matter", "point", "state", "time", "way", "work"
]

st.subheader("Input Kalimat")
sentence = st.text_area(
    "Masukkan kalimat yang mengandung kata ambigu",
    height=100,
    placeholder="Contoh: The case was discussed in court yesterday.",
    label_visibility="collapsed"
)

target_word = st.selectbox("Pilih kata ambigu", ambiguous_words)

# ==================================================
# PREDIKSI
# ==================================================
if st.button("Prediksi Sense", use_container_width=True):
    if not sentence.strip():
        st.warning("Masukkan kalimat terlebih dahulu.")
        st.stop()

    tokens = sentence.lower().split()
    if target_word not in tokens:
        st.warning(f"Kata '{target_word}' tidak ditemukan dalam kalimat.")
        st.stop()

    idx = tokens.index(target_word)

    try:
        with st.spinner("Memproses embedding dan pencarian sense..."):
            inst_vecs = encoder.token_embeddings([tokens])[0][0]
            word_vec = inst_vecs[idx][1]
            word_vec = word_vec / np.linalg.norm(word_vec)

            preds = senses_vsm.most_similar_vec(word_vec, topn=None)
            preds = [(sense, score) for sense, score in preds
                     if sense.lower().startswith(target_word.lower() + ".")]

            if not preds:
                st.error(f"Tidak ditemukan sense untuk kata '{target_word}'")
                st.stop()

            best_sense, best_score = preds[0]

        st.markdown("---")
        st.subheader("Hasil Prediksi")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">SENSE TERPILIH</div>
                <div class="result-value">{best_sense}</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">SIMILARITY SCORE</div>
                <div class="result-value">{best_score:.4f}</div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")
        st.progress(min(float(best_score), 1.0))
        st.caption("Nilai similarity menunjukkan tingkat kemiripan antara embedding kontekstual dengan sense vector yang tersimpan.")

    except Exception as e:
        st.error(f"Error: {str(e)}")

# ==================================================
# FOOTER
# ==================================================
st.markdown("""
<div class="footer">
    Word Sense Disambiguation • BERT Base Uncased • 1-NN • SemCor-13
</div>
""", unsafe_allow_html=True)