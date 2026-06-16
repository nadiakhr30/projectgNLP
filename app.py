import streamlit as st
import numpy as np
from nlm_encoder import TransformerEncoder
from vectorspace import VSM

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Word Sense Disambiguation",
    page_icon="⚙️",  # optional, bisa dihapus
    layout="wide"
)

# ==================================================
# CUSTOM CSS (BERWARNA DAN MODERN)
# ==================================================
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,300;14..32,400;14..32,500;14..32,600;14..32,700;14..32,800&display=swap" rel="stylesheet">
    <style>
        * {
            font-family: 'Inter', sans-serif;
        }

        /* Background gradien lembut */
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(145deg, #f0f4fe 0%, #e8edf9 100%);
        }

        .main .block-container {
            max-width: 1200px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        section[data-testid="stSidebar"] {
            background: rgba(255,255,255,0.85);
            backdrop-filter: blur(6px);
            border-right: 1px solid rgba(255,255,255,0.3);
            box-shadow: 2px 0 20px rgba(0,0,0,0.02);
        }

        /* Hero dengan gradien dan bayangan */
        .hero {
            background: linear-gradient(135deg, #ffffff 0%, #f2f7ff 100%);
            border-radius: 32px;
            padding: 2.4rem 3rem;
            margin-bottom: 2.5rem;
            border: 1px solid rgba(255,255,255,0.6);
            box-shadow: 0 20px 40px -16px rgba(30,64,175,0.15), 0 0 0 1px rgba(59,130,246,0.05);
            transition: all 0.3s;
        }
        .hero:hover {
            box-shadow: 0 25px 50px -18px rgba(30,64,175,0.2);
        }

        .hero-label {
            display: inline-block;
            background: linear-gradient(90deg, #3b82f6, #6366f1);
            color: white;
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 0.8px;
            padding: 0.3rem 1.2rem;
            border-radius: 40px;
            text-transform: uppercase;
            margin-bottom: 1rem;
        }

        .hero-title {
            font-size: 2.8rem;
            font-weight: 800;
            background: linear-gradient(135deg, #0f172a, #2563eb);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            line-height: 1.2;
            margin-bottom: 0.8rem;
        }

        .hero-sub {
            font-size: 1rem;
            line-height: 1.6;
            color: #1e293b;
            max-width: 85%;
            margin-bottom: 1.2rem;
        }

        .hero-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 0.6rem;
        }
        .tag {
            background: rgba(59,130,246,0.08);
            border: 1px solid rgba(59,130,246,0.15);
            padding: 0.2rem 1rem;
            border-radius: 30px;
            font-size: 0.75rem;
            font-weight: 500;
            color: #1e40af;
            letter-spacing: 0.2px;
        }
        .tag:hover {
            background: rgba(59,130,246,0.15);
        }

        /* Stat card dengan warna aksen */
        .stat-card {
            background: white;
            border-radius: 24px;
            padding: 1.8rem 1rem;
            text-align: center;
            border: 1px solid #e2e8f0;
            box-shadow: 0 4px 12px rgba(0,0,0,0.02);
            transition: 0.25s;
            position: relative;
            overflow: hidden;
        }
        .stat-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        }
        .stat-card:hover {
            transform: translateY(-6px);
            box-shadow: 0 20px 30px -12px rgba(59,130,246,0.15);
            border-color: #b9d0f0;
        }
        .stat-value {
            font-size: 2.2rem;
            font-weight: 800;
            color: #0f172a;
            line-height: 1.3;
        }
        .stat-label {
            font-size: 0.7rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #475569;
            margin-top: 0.3rem;
        }
        .stat-desc {
            font-size: 0.65rem;
            color: #94a3b8;
            margin-top: 0.1rem;
        }

        /* Result card dengan gradien */
        .result-card {
            background: white;
            border-radius: 24px;
            padding: 2rem;
            text-align: center;
            box-shadow: 0 8px 24px rgba(0,0,0,0.04);
            border: 1px solid #eef2f6;
            transition: 0.2s;
        }
        .result-card:hover {
            border-color: #93b4e8;
        }
        .result-card-sense {
            border-left: 6px solid #3b82f6;
        }
        .result-card-score {
            border-left: 6px solid #8b5cf6;
        }
        .result-label {
            font-size: 0.7rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #64748b;
        }
        .result-value {
            font-size: 1.8rem;
            font-weight: 800;
            color: #0f172a;
            margin-top: 0.5rem;
            word-break: break-word;
        }

        /* Form elements */
        .stTextArea textarea, .stSelectbox > div > div {
            border-radius: 16px !important;
            border: 1px solid #d1d9e6 !important;
            transition: 0.2s;
        }
        .stTextArea textarea:focus, .stSelectbox > div > div:focus-within {
            border-color: #3b82f6 !important;
            box-shadow: 0 0 0 3px rgba(59,130,246,0.2) !important;
        }

        /* Tombol dengan gradien */
        .stButton > button {
            width: 100%;
            height: 54px;
            border: none;
            border-radius: 16px;
            background: linear-gradient(95deg, #2563eb, #7c3aed);
            color: white;
            font-weight: 700;
            font-size: 1rem;
            transition: 0.25s;
            box-shadow: 0 4px 14px rgba(37,99,235,0.25);
        }
        .stButton > button:hover {
            transform: translateY(-3px);
            box-shadow: 0 12px 28px -8px rgba(37,99,235,0.4);
            background: linear-gradient(95deg, #1d4ed8, #6d28d9);
        }

        .stProgress > div > div > div {
            background: linear-gradient(90deg, #2563eb, #8b5cf6);
            border-radius: 20px;
        }

        .footer {
            text-align: center;
            margin-top: 3rem;
            padding-top: 1.2rem;
            border-top: 1px solid #dce3ed;
            font-size: 0.7rem;
            color: #64748b;
        }

        h1, h2, h3, h4 {
            color: #0f172a;
            font-weight: 600;
        }

        hr {
            margin: 1.5rem 0;
            border-color: #dce3ed;
        }

        .sidebar-heading {
            font-weight: 700;
            font-size: 1.1rem;
            letter-spacing: -0.2px;
            color: #0f172a;
            border-bottom: 2px solid #3b82f6;
            padding-bottom: 0.5rem;
            margin-bottom: 1.2rem;
        }

        .info-block {
            font-size: 0.85rem;
            line-height: 1.6;
            color: #1e293b;
        }
        .info-block strong {
            color: #0f172a;
        }
        .info-block code {
            background: #eef2ff;
            padding: 0.1rem 0.4rem;
            border-radius: 6px;
            font-size: 0.8rem;
            color: #1e40af;
        }
        .sidebar-example {
            background: #f8fafc;
            border-radius: 12px;
            padding: 0.8rem 1rem;
            margin-top: 0.5rem;
            border-left: 3px solid #3b82f6;
            font-size: 0.8rem;
            color: #1e293b;
        }

        /* Warna untuk tag di hero */
        .tag-blue { background: #dbeafe; color: #1e40af; }
        .tag-purple { background: #ede9fe; color: #5b21b6; }
        .tag-green { background: #d1fae5; color: #065f46; }
        .tag-rose { background: #ffe4e6; color: #9f1239; }
    </style>
""", unsafe_allow_html=True)

# ==================================================
# HEADER (DENGAN WARNA GRADIEN)
# ==================================================
st.markdown("""
<div class="hero">
    <div class="hero-label">✦ WORD SENSE DISAMBIGUATION</div>
    <div class="hero-title">Menentukan Makna Kata Ambigu<br>dalam Konteks Kalimat</div>
    <div class="hero-sub">
        Sistem berbasis <strong>BERT Base Uncased</strong> dan metode <strong>1-Nearest Neighbor (1-NN)</strong>
        untuk memetakan kata ambigu ke sense yang paling sesuai berdasarkan 
        embedding kontekstual dari dataset <strong>SemCor-13</strong>.
    </div>
    <div class="hero-tags">
        <span class="tag tag-blue">BERT Base Uncased</span>
        <span class="tag tag-purple">1-NN Classifier</span>
        <span class="tag tag-green">SemCor-13</span>
        <span class="tag tag-rose">Contextual Embedding</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ==================================================
# SIDEBAR (BERWARNA)
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
    Sense dengan <strong>cosine similarity</strong> tertinggi dipilih sebagai makna.
    <br><br>
    <strong>Contoh kalimat</strong>
    <div class="sidebar-example">
        • The case was presented to the judge.<br>
        • She wore a watch on her wrist.<br>
        • Time flies like an arrow.
    </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption("© 2025 • WSD System")

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
    senses_vsm = VSM("vectors/semcor13_test.txt", normalize=True)
    return encoder, senses_vsm

encoder, senses_vsm = load_model()

# ==================================================
# STATISTIK (BERWARNA)
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
# INPUT
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
            <div class="result-card result-card-sense">
                <div class="result-label">SENSE TERPILIH</div>
                <div class="result-value">{best_sense}</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="result-card result-card-score">
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