import streamlit as st
import numpy as np
from nlm_encoder import TransformerEncoder
from vectorspace import VSM

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Word Sense Disambiguation",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# CUSTOM CSS (DITINGKATKAN)
# ==================================================
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,300;14..32,400;14..32,600;14..32,700;14..32,800&display=swap" rel="stylesheet">
    <style>
        * {
            font-family: 'Inter', sans-serif;
        }

        /* Latar belakang utama */
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #f5f7fc 0%, #eef2f6 100%);
        }

        /* Kontainer utama */
        .main .block-container {
            max-width: 1300px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: rgba(255,255,255,0.95);
            backdrop-filter: blur(2px);
            border-right: none;
            box-shadow: 4px 0 20px rgba(0,0,0,0.02);
        }

        /* ========== HERO SECTION BARU ========== */
        .hero-new {
            background: linear-gradient(120deg, #ffffff 0%, #f8fafc 100%);
            border-radius: 2rem;
            padding: 2rem 2.5rem;
            margin-bottom: 2.5rem;
            border: 1px solid rgba(255,255,255,0.5);
            box-shadow: 0 20px 35px -12px rgba(0,0,0,0.08), 0 0 0 1px rgba(0,0,0,0.02);
            transition: all 0.3s ease;
        }
        .hero-new:hover {
            transform: translateY(-3px);
            box-shadow: 0 25px 40px -14px rgba(0,0,0,0.12);
        }
        .hero-badge {
            display: inline-block;
            background: #e0e7ff;
            color: #1e40af;
            font-size: 0.75rem;
            font-weight: 600;
            padding: 0.2rem 0.8rem;
            border-radius: 30px;
            margin-bottom: 1rem;
            letter-spacing: 0.3px;
        }
        .hero-title {
            font-size: 3rem;
            font-weight: 800;
            background: linear-gradient(135deg, #0f172a, #2563eb);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            margin-bottom: 0.75rem;
            line-height: 1.2;
        }
        .hero-sub {
            color: #334155;
            font-size: 1rem;
            line-height: 1.6;
            max-width: 85%;
            margin-bottom: 1rem;
        }
        .hero-meta {
            display: flex;
            gap: 1.5rem;
            margin-top: 1.2rem;
            flex-wrap: wrap;
        }
        .hero-meta-item {
            background: #f1f5f9;
            border-radius: 40px;
            padding: 0.3rem 1rem;
            font-size: 0.8rem;
            font-weight: 500;
            color: #1e293b;
        }

        /* Stat Card */
        .stat-card {
            background: white;
            border-radius: 1.5rem;
            padding: 1.5rem;
            text-align: center;
            transition: all 0.25s ease;
            border: 1px solid #eef2ff;
            box-shadow: 0 4px 12px rgba(0,0,0,0.02);
        }
        .stat-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 20px 25px -12px rgba(0,0,0,0.1);
            border-color: #cbd5e1;
        }
        .stat-icon {
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }
        .stat-title {
            color: #475569;
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .stat-value {
            color: #1e293b;
            font-size: 2.2rem;
            font-weight: 800;
            margin-top: 0.25rem;
        }

        /* Result Card */
        .result-card {
            background: white;
            border-radius: 1.5rem;
            padding: 1.8rem;
            text-align: center;
            border-left: 6px solid #2563eb;
            box-shadow: 0 12px 28px -8px rgba(0,0,0,0.08);
            transition: 0.2s;
        }
        .result-title {
            color: #64748b;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            font-weight: 600;
        }
        .result-value {
            color: #0f172a;
            font-size: 1.7rem;
            font-weight: 800;
            margin-top: 0.5rem;
            word-break: break-word;
        }

        /* Form elements */
        .stTextArea textarea, .stSelectbox > div > div {
            border-radius: 1rem !important;
            border: 1px solid #e2e8f0 !important;
            transition: 0.2s;
        }
        .stTextArea textarea:focus, .stSelectbox > div > div:focus-within {
            border-color: #2563eb !important;
            box-shadow: 0 0 0 2px rgba(37,99,235,0.2) !important;
        }

        /* Tombol */
        .stButton > button {
            width: 100%;
            height: 52px;
            border: none;
            border-radius: 1rem;
            background: linear-gradient(95deg, #2563eb, #1d4ed8);
            color: white;
            font-weight: 700;
            font-size: 1rem;
            transition: 0.2s;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 20px -12px #1e3a8a;
            background: linear-gradient(95deg, #1d4ed8, #1e3a8a);
        }

        /* Progress bar */
        .stProgress > div > div > div {
            background: linear-gradient(90deg, #2563eb, #60a5fa);
            border-radius: 20px;
        }

        /* Footer */
        .footer {
            text-align: center;
            margin-top: 3rem;
            padding-top: 1.5rem;
            border-top: 1px solid #e2e8f0;
            font-size: 0.8rem;
            color: #64748b;
        }

        /* Sidebar text */
        .sidebar-title {
            font-weight: 700;
            font-size: 1.1rem;
            margin-bottom: 1rem;
            color: #0f172a;
        }
        hr {
            margin: 1rem 0;
        }

        @media (max-width: 768px) {
            .hero-title { font-size: 2rem; }
            .hero-sub { max-width: 100%; }
            .hero-new { padding: 1.5rem; }
        }
    </style>
""", unsafe_allow_html=True)

# ==================================================
# HEADER YANG DIPERBAIKI
# ==================================================
st.markdown("""
<div class="hero-new">
    <div class="hero-badge">
        🔬 NLP Systems
    </div>
    <div class="hero-title">
        Word Sense Disambiguation
    </div>
    <div class="hero-sub">
        Menentukan makna kata ambigu dalam kalimat menggunakan <strong>BERT Base Uncased</strong> 
        dan metode <strong>1-Nearest Neighbor (1-NN)</strong> berbasis dataset SemCor-13.
    </div>
    <div class="hero-meta">
        <span class="hero-meta-item">🧠 BERT Base Uncased</span>
        <span class="hero-meta-item">📊 1-NN Classifier</span>
        <span class="hero-meta-item">📚 SemCor-13</span>
        <span class="hero-meta-item">⚡ Contextual Embedding</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ==================================================
# SIDEBAR (DIPERINDAH)
# ==================================================
with st.sidebar:
    st.markdown('<div class="sidebar-title">📌 Informasi Model</div>', unsafe_allow_html=True)
    st.markdown("""
    **🤖 Model**  
    `BERT Base Uncased`

    **📐 Metode**  
    `1-Nearest Neighbor (1-NN)`

    **🗂️ Dataset**  
    `SemCor-13 (13 kata ambigu)`

    ---
    
    **💡 Cara kerja**  
    Embedding kontekstual kata ambigu dibandingkan dengan *sense vector* yang telah dilatih pada dataset.  
    Sense dengan *cosine similarity* tertinggi dipilih sebagai makna.
    
    ---
    
    **🧪 Contoh kalimat**  
    - *The case was presented to the judge.*  
    - *She wore a watch on her wrist.*  
    - *Time flies like an arrow.*
    """)
    
    st.markdown("---")
    st.caption("© 2025 | WSD System")

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
# STATISTIK DENGAN IKON
# ==================================================
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-icon">📖</div>
        <div class="stat-title">Jumlah Kata Ambigu</div>
        <div class="stat-value">13</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-icon">🧠</div>
        <div class="stat-title">Model</div>
        <div class="stat-value">BERT</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-icon">🎯</div>
        <div class="stat-title">Metode</div>
        <div class="stat-value">1-NN</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==================================================
# INPUT KALIMAT & KATA AMBIGU
# ==================================================
ambiguous_words = [
    "case", "face", "form", "head", "interest", "life", "light",
    "matter", "point", "state", "time", "way", "work"
]

st.subheader("✍️ Input Kalimat")
sentence = st.text_area(
    "Masukkan kalimat yang mengandung kata ambigu",
    height=110,
    placeholder="Contoh: The case was discussed in court yesterday.",
    label_visibility="collapsed"
)

target_word = st.selectbox("🔍 Pilih kata ambigu", ambiguous_words)

# ==================================================
# PREDIKSI
# ==================================================
if st.button("🚀 Prediksi Sense", use_container_width=True):
    if sentence.strip() == "":
        st.warning("⚠️ Masukkan kalimat terlebih dahulu.")
        st.stop()
    
    tokens = sentence.lower().split()
    if target_word not in tokens:
        st.warning(f"❌ Kata '{target_word}' tidak ditemukan dalam kalimat.")
        st.stop()
    
    idx = tokens.index(target_word)
    
    try:
        with st.spinner("🔄 Memproses embedding dan mencari sense terdekat..."):
            inst_vecs = encoder.token_embeddings([tokens])[0][0]
            word_vec = inst_vecs[idx][1]
            word_vec = word_vec / np.linalg.norm(word_vec)
            preds = senses_vsm.most_similar_vec(word_vec, topn=None)
            preds = [(sense, score) for sense, score in preds
                     if sense.lower().startswith(target_word.lower() + ".")]
            if len(preds) == 0:
                st.error(f"❌ Tidak ditemukan sense untuk kata '{target_word}'")
                st.stop()
            best_sense, best_score = preds[0]
        
        st.markdown("---")
        st.subheader("📊 Hasil Prediksi")
        colA, colB = st.columns(2)
        with colA:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-title">🎯 Sense Terpilih</div>
                <div class="result-value">{best_sense}</div>
            </div>
            """, unsafe_allow_html=True)
        with colB:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-title">📈 Similarity Score</div>
                <div class="result-value">{best_score:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.write("")
        st.progress(min(float(best_score), 1.0))
        st.caption("✨ *Similarity mendekati 1 berarti sangat mirip dengan sense yang tersimpan.*")
    
    except Exception as e:
        st.error(f"Terjadi kesalahan: {str(e)}")

# ==================================================
# FOOTER
# ==================================================
st.markdown("""
<div class="footer">
    Dibangun dengan Streamlit • BERT Base Uncased • 1-NN Word Sense Disambiguation
</div>
""", unsafe_allow_html=True)